"""
SSL methodology wrappers implementing specific view generation, augmentation, spatial masking, and loss computation
for CroSSL, COCOA, SimCLR, and CPC frameworks.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..models.backbone import SSSLBackbone
from ..losses.mvcl_losses import COCOALoss, VICRegLoss
from ..losses.baseline_losses import NTXentLoss, InfoNCELoss


class BaseSSLMethod(nn.Module):
    """Abstract base class for self-supervised learning methods."""
    def __init__(self, backbone: SSSLBackbone):
        super().__init__()
        self.backbone = backbone

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CroSSLMethod(BaseSSLMethod):
    """
    CroSSL Method: Multi-view contrastive learning with latent spatial masking and VICReg loss.
    Optimizes both individual view encoders and cross-view aggregator simultaneously during pretraining.
    """
    def __init__(self, backbone: SSSLBackbone, mask_prob: float = 0.3, **loss_kwargs):
        super().__init__(backbone)
        self.mask_prob = mask_prob
        self.criterion = VICRegLoss(**loss_kwargs)
        self.num_views = backbone.encoder.num_views

    def _generate_random_mask(self, batch_size: int, num_views: int, device: torch.device) -> torch.Tensor:
        # Create random Bernoulli mask with keep probability (1 - mask_prob)
        keep_prob = 1.0 - self.mask_prob
        mask = torch.bernoulli(torch.full((batch_size, num_views), keep_prob, device=device))
        
        # Ensure at least one sensor view remains unmasked per sample in batch
        all_zeros = (mask.sum(dim=1) == 0)
        if all_zeros.any():
            random_idx = torch.randint(0, num_views, (all_zeros.sum().item(),), device=device)
            mask[all_zeros, random_idx] = 1.0
        return mask

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Multi-view IMU tensor of shape (B, N, C, T).
        Returns:
            Scalar VICReg pretraining loss.
        """
        B, N, C, T = x.shape
        # 1. Encode into multi-view representations
        view_embeds = self.backbone.encoder(x) # (B, N, 64)
        
        # 2. Generate two independent spatial masks across sensor views
        mask1 = self._generate_random_mask(B, N, x.device)
        mask2 = self._generate_random_mask(B, N, x.device)
        
        # Ensure masks are not completely identical for all samples when possible
        if N > 1 and torch.all(mask1 == mask2):
            mask2 = torch.roll(mask2, shifts=1, dims=1)
            
        # 3. Aggregate masked features
        z1 = self.backbone.aggregator(view_embeds, mask=mask1) # (B, 256)
        z2 = self.backbone.aggregator(view_embeds, mask=mask2) # (B, 256)
        
        return self.criterion(z1, z2)


class COCOAMethod(BaseSSLMethod):
    """
    COCOA Method: Cross-modality contrastive alignment loss on individual sensor view embeddings.
    """
    def __init__(self, backbone: SSSLBackbone, **loss_kwargs):
        super().__init__(backbone)
        self.criterion = COCOALoss(**loss_kwargs)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        view_embeds = self.backbone.encoder(x) # (B, N, 64)
        return self.criterion(view_embeds)


class SimCLRMethod(BaseSSLMethod):
    """
    SimCLR Method: Conventional non-MVCL contrastive framework using Gaussian noise and temporal scaling.
    """
    def __init__(self, backbone: SSSLBackbone, noise_std: float = 0.1, **loss_kwargs):
        super().__init__(backbone)
        self.noise_std = noise_std
        self.criterion = NTXentLoss(**loss_kwargs)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Additive Gaussian noise augmentation + Random scaling factor
        scale = torch.empty((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(0.8, 1.2)
        noise = torch.randn_like(x) * self.noise_std
        return (x * scale) + noise

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._augment(x)
        x2 = self._augment(x)
        
        z1 = self.backbone(x1) # (B, 256)
        z2 = self.backbone(x2) # (B, 256)
        return self.criterion(z1, z2)


class CPCMethod(BaseSSLMethod):
    """
    CPC Method: Contrastive Predictive Coding predicting future time segments from past contextual embeddings.
    """
    def __init__(self, backbone: SSSLBackbone, **loss_kwargs):
        super().__init__(backbone)
        agg_dim = backbone.aggregator.fc2.out_features
        # Autoregressive / linear projection to predict future embedding from past embedding
        self.predictor = nn.Linear(agg_dim, agg_dim)
        self.criterion = InfoNCELoss(**loss_kwargs)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, T = x.shape
        half_T = max(8, T // 2)
        
        x_past = x[:, :, :, :half_T]
        x_future = x[:, :, :, half_T:]
        
        z_past = self.backbone(x_past)     # (B, 256)
        z_future = self.backbone(x_future) # (B, 256)
        
        c_proj = self.predictor(z_past)
        return self.criterion(c_proj, z_future)


def get_ssl_method_trainer(method_name: str, backbone: SSSLBackbone, **kwargs) -> BaseSSLMethod:
    """Factory function returning the corresponding SSL method implementation."""
    name = method_name.lower().replace("-", "").replace("_", "")
    if "crossl" in name:
        return CroSSLMethod(backbone, **kwargs)
    elif "cocoa" in name:
        return COCOAMethod(backbone, **kwargs)
    elif "simclr" in name:
        return SimCLRMethod(backbone, **kwargs)
    elif "cpc" in name:
        return CPCMethod(backbone, **kwargs)
    elif "supervised" in name or "baseline" in name:
        raise ValueError("Supervised baseline does not utilize SSL pretraining.")
    else:
        raise ValueError(f"Unknown SSL method: '{method_name}'")
