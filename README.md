# Self-Flow ImageNet Inference

This folder contains inference code for generating images with our Self-Flow trained diffusion model on ImageNet 256×256.

## Overview

**Self-Flow** (Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis) is a training method for diffusion transformers that combines the flow matching objective with a self-supervised feature reconstruction objective.

This inference code allows you to:

1. Load a Self-Flow checkpoints (pretrained on ImageNet 256x256)
2. Generate 50,000 images for FID evaluation

The generated samples can be evaluated using the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate 50k samples (multi-GPU recommended)

```bash
torchrun --nnodes=1 --nproc_per_node=8 sample.py \
    --ckpt checkpoints/selfflow_imagenet256.pt \
    --output-dir ./samples \
    --num-fid-samples 50000
```

### Single GPU

```bash
python sample.py \
    --ckpt checkpoints/selfflow_imagenet256.pt \
    --output-dir ./samples \
    --num-fid-samples 50000 \
    --batch-size 64
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | required | Path to model checkpoint |
| `--output-dir` | `./samples` | Output directory for generated samples |
| `--num-fid-samples` | `50000` | Number of samples to generate |
| `--batch-size` | `64` | Batch size per GPU |
| `--num-steps` | `250` | Number of diffusion sampling steps |
| `--mode` | `SDE` | Sampling mode: `SDE` or `ODE` |
| `--seed` | `31` | Random seed for reproducibility |
| `--cfg-scale` | `1.0` | Classifier-free guidance scale (1.0 = no guidance, as used in paper) |

## Evaluation

The generated `.npz` file can be used with the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, IS, Precision, and Recall.

### Download Reference Statistics

```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

### Run Evaluation

```bash
python evaluator.py \
    VIRTUAL_imagenet256_labeled.npz \
    ./samples/samples_50000.npz
```

## Model Architecture

The Self-Flow model is based on SiT-XL/2 with the following specifications

A key architectural modification is **per-token timestep conditioning**, which allows each token to have a different noise level during training.

## Project Structure

```
ImageNet_inference/
├── sample.py           # Main sampling script
├── checkpoints/        # Place model checkpoints here
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── src/                # Model and sampling implementations
    ├── model.py        # SelfFlowPerTokenDiT model
    ├── sampling.py     # Diffusion sampling utilities
    └── utils.py        # Position encoding utilities
```

## Training Details

The model was trained using the following configuration:

- **Model**: SiT-XL/2 with per-token timestep conditioning
- **Training**: Self-Flow with per-token masking (25% mask ratio)
- **Optimizer**: AdamW with gradient clipping (max_norm=1)
- **Mixed precision**: BFloat16
- **Self-distillation**: Teacher at layer 20 (EMA), student at layer 8

## Citation

If you use this code, please cite:

```bibtex
@article{selfflow2025,
  title={Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis},
  author={...},
  year={2026}
}
```

## Acknowledgments

This code builds upon:
- [REPA](https://github.com/sihyun-yu/REPA) - Representation Alignment for Generation
- [SiT](https://github.com/willisma/SiT) - Scalable Interpolant Transformers