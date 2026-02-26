#!/usr/bin/env python3
"""
Sample images from a trained Self-Flow diffusion model.

Usage:
    # Single GPU
    python sample.py --ckpt path/to/checkpoint.pt --output-dir ./samples

    # Multi-GPU with torchrun
    torchrun --nnodes=1 --nproc_per_node=8 sample.py --ckpt path/to/checkpoint.pt

This script generates images for FID evaluation, outputting an NPZ file
compatible with the ADM evaluation suite.
"""

import os
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from einops import rearrange

# Import from local src/ folder
from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop
from src.utils import batched_prc_img, scattercat


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = 0
        torch.cuda.set_device(device)
    
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_npz_from_samples(samples, output_path):
    """Save samples to NPZ file for ADM evaluation."""
    samples = np.stack(samples, axis=0)
    np.savez(output_path, arr_0=samples)
    print(f"Saved {len(samples)} samples to {output_path}")


def load_vae(device, dtype=torch.bfloat16):
    """Load the SD-VAE for decoding latents to images."""
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    
    # VAE scaling factors
    scale_factor = 0.18215
    shift_factor = 0.0
    
    return vae, scale_factor, shift_factor


def decode_latents(vae, latents, scale_factor, shift_factor):
    """Decode latents to images using the VAE."""
    latents = latents / scale_factor + shift_factor
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=vae_dtype)
    
    with torch.no_grad():
        images = vae.decode(latents).sample
    
    images = (images.float() + 1) / 2
    images = images.clamp(0, 1)
    
    return images


def load_model(ckpt_path, device):
    """Load the Self-Flow model from checkpoint."""
    print(f"Loading model from {ckpt_path}")
    
    # Create model with DiT-XL/2 settings
    config = dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )
    model = SelfFlowPerTokenDiT(**config)
    
    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    
    # Keep model in float32 for weights, autocast handles precision
    model = model.to(device=device)
    model.eval()
    
    return model


@torch.no_grad()
def sample_batch(
    model,
    vae,
    scale_factor,
    shift_factor,
    batch_size,
    class_labels,
    num_steps=250,
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    mode="SDE",
    device="cuda",
    seed=None,
):
    """Sample a batch of images."""
    if seed is not None:
        torch.manual_seed(seed)
    
    latent_size = 32
    latent_channels = 4
    patch_size = 2
    
    # Sample noise in bfloat16 for mixed precision
    noise = torch.randn(
        batch_size, latent_channels, latent_size, latent_size,
        device=device, dtype=torch.bfloat16
    )
    
    # Patchify noise: (B, C, H, W) -> (B, C*P*P, H/P, W/P)
    noise_patched = rearrange(
        noise,
        "b c (h p1) (w p2) -> b (c p1 p2) h w",
        p1=patch_size, p2=patch_size
    )
    
    # Convert to token format
    x, x_ids = batched_prc_img(noise_patched.cpu())
    x = x.to(device=device)
    x_ids = x_ids.to(device)
    
    # Prepare for CFG if needed
    if cfg_scale > 1.0:
        x = torch.cat([x, x], dim=0)
        x_ids = torch.cat([x_ids, x_ids], dim=0)
        null_labels = torch.full_like(class_labels, 1000)
        class_labels = torch.cat([null_labels, class_labels], dim=0)
    
    # Run denoising with bfloat16 autocast
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        samples = denoise_loop(
            model=model,
            num_steps=num_steps,
            cfg_scale=cfg_scale if cfg_scale > 1.0 else None,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            mode=mode,
            x=x,
            x_ids=x_ids,
            vector=class_labels,
        )
    
    # Extract conditional output if CFG was used
    if cfg_scale > 1.0:
        samples = samples[batch_size:]
        x_ids = x_ids[batch_size:]
    
    # Convert back to image format
    samples = scattercat(samples, x_ids)
    
    # Unpatchify: (B, C*P*P, H/P, W/P) -> (B, C, H, W)
    samples = rearrange(
        samples,
        "b (c p1 p2) h w -> b c (h p1) (w p2)",
        p1=patch_size, p2=patch_size, c=latent_channels
    )
    
    # Decode with VAE
    images = decode_latents(vae, samples, scale_factor, shift_factor)
    
    # Convert to numpy [0, 255]
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = (images * 255).astype(np.uint8)
    
    return images


def main():
    parser = argparse.ArgumentParser(description="Sample images from Self-Flow model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./samples", help="Output directory")
    parser.add_argument("--num-fid-samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--num-steps", type=int, default=250, help="Number of diffusion steps")
    parser.add_argument("--mode", type=str, default="SDE", choices=["SDE", "ODE"], help="Sampling mode")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    parser.add_argument("--save-images", action="store_true", default=True, help="Save individual PNG images")
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    # CFG options (not used in paper results)
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--guidance-low", type=float, default=0.0, help="Lower guidance bound")
    parser.add_argument("--guidance-high", type=float, default=0.7, help="Upper guidance bound")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    device = f"cuda:{device}"
    
    if rank == 0:
        print(f"Running on {world_size} GPU(s)")
        print(f"Generating {args.num_fid_samples} samples")
        print(f"Mode: {args.mode}, Steps: {args.num_steps}, CFG: {args.cfg_scale}")
    
    # Set seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_images:
            (output_dir / "images").mkdir(exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Load models
    model = load_model(args.ckpt, device)
    vae, scale_factor, shift_factor = load_vae(device)
    
    # Calculate samples per GPU
    total_samples = args.num_fid_samples
    samples_per_gpu = math.ceil(total_samples / world_size)
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, total_samples)
    my_samples = end_idx - start_idx
    
    if rank == 0:
        print(f"Each GPU will generate ~{samples_per_gpu} samples")
    
    # Generate samples
    all_samples = []
    all_labels = []
    
    num_batches = math.ceil(my_samples / args.batch_size)
    pbar = tqdm(range(num_batches), desc=f"GPU {rank}", disable=rank != 0)
    
    for batch_idx in pbar:
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, my_samples)
        batch_size = batch_end - batch_start
        
        # Random class labels
        class_labels = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Unique seed for this batch
        batch_seed = args.seed + rank * 100000 + batch_idx
        
        images = sample_batch(
            model=model,
            vae=vae,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
            batch_size=batch_size,
            class_labels=class_labels,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            mode=args.mode,
            device=device,
            seed=batch_seed,
        )
        
        all_samples.append(images)
        all_labels.extend(class_labels.cpu().numpy())
        
        if args.save_images and rank == 0:
            for i, img in enumerate(images):
                global_idx = start_idx + batch_start + i
                Image.fromarray(img).save(output_dir / "images" / f"{global_idx:06d}.png")
    
    all_samples = np.concatenate(all_samples, axis=0)
    
    # Gather from all GPUs using file-based approach
    if world_size > 1:
        rank_npz = output_dir / f"samples_rank{rank}.npz"
        np.savez(rank_npz, arr_0=all_samples)
        dist.barrier()
        
        if rank == 0:
            gathered = []
            for r in range(world_size):
                r_path = output_dir / f"samples_rank{r}.npz"
                r_data = np.load(r_path)['arr_0']
                gathered.append(r_data)
                r_path.unlink()
            all_samples = np.concatenate(gathered, axis=0)
    
    # Save NPZ
    if rank == 0:
        all_samples = all_samples[:args.num_fid_samples]
        npz_path = output_dir / f"samples_{len(all_samples)}.npz"
        create_npz_from_samples(list(all_samples), npz_path)
        
        print(f"\nDone! Generated {len(all_samples)} samples")
        print(f"NPZ: {npz_path}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
