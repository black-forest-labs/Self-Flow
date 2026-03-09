<h1 align="center"> Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis (Self-Flow)
</h1>

<p align="center">
  <strong>Hila Chefer*</strong> · <strong>Patrick Esser*</strong> <br> <strong>Dominik Lorenz</strong> · <strong>Dustin Podell</strong> · <strong>Vikash Raja</strong> · <strong>Vinh Tong</strong> · <strong>Antonio Torralba</strong> · <strong>Robin Rombach</strong>  <br>
  <em>Black Forest Labs</em>
</p>

<p align="center">
  <a href="https://bfl.ai/research/self-flow">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Website">
  </a>
  <a href="https://cdn.sanity.io/files/2gpum2i6/production/3264c9e34dfcbbf57e57ca7efee0a3e83d311a83.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper PDF">
  </a>
    <a href="https://arxiv.org/abs/2603.06507">
    <img src="https://img.shields.io/badge/arXiv-2603.06507-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
</p>

This folder contains inference code for generating images with our [Self-Flow](https://bfl.ai/research/self-flow) trained diffusion model on ImageNet 256×256.

## Overview

**Self-Flow** (Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis) is a training framework that combines the flow matching objective with a self-supervised feature reconstruction objective.

This inference code allows you to:

1. Load a Self-Flow checkpoints (pretrained on ImageNet 256x256)
2. Generate 50,000 images for FID evaluation

The generated samples can be evaluated using the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

### Download Checkpoint

```
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Hila/Self-Flow',
    filename='selfflow_imagenet256.pt',
    local_dir='./checkpoints'
)
print('Downloaded!')
"
```

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
    ./samples/samples_50000.npz ./samples
```

## Model Architecture

The Self-Flow model is based on SiT-XL/2 with the following specifications

A key architectural modification is **per-token timestep conditioning**, which allows each token to have a different noise level during training.

## Project Structure

```
Self-Flow/
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

## Acknowledgments

This code builds upon:
- [REPA](https://github.com/sihyun-yu/REPA) - Representation Alignment for Generation
- [SiT](https://github.com/willisma/SiT) - Scalable Interpolant Transformers

## BibTeX
If you use this work, please cite: 

```bibtex
@article{CheferEsser2026selfflow,
      title={Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis}, 
      author={Hila Chefer and Patrick Esser and Dominik Lorenz and Dustin Podell and Vikash Raja and Vinh Tong and Antonio Torralba and Robin Rombach},
      journal = {arXiv preprint arXiv:2603.06507},
      year={2026},
}
```
