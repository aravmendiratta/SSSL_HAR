"""
Self-supervised contrastive learning loss functions for IMU HAR representation learning.
"""

from .mvcl_losses import COCOALoss, VICRegLoss
from .baseline_losses import NTXentLoss, InfoNCELoss

__all__ = [
    "COCOALoss",
    "VICRegLoss",
    "NTXentLoss",
    "InfoNCELoss",
]
