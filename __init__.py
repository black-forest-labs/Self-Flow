"""
Self-Flow ImageNet Inference Package.

This package provides standalone inference code for generating images with
Self-Flow trained diffusion models on ImageNet 256×256.

Self-Flow: Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis
"""

from .src.model import SelfFlowPerTokenDiT
from .src.sampling import denoise_loop
from .src.utils import batched_prc_img, scattercat

__all__ = [
    "SelfFlowPerTokenDiT",
    "denoise_loop",
    "batched_prc_img",
    "scattercat",
]
