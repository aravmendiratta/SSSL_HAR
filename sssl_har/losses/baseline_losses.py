"""
Baseline SSL contrastive loss functions: SimCLR (NT-Xent) and CPC (InfoNCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for SimCLR framework.
    Operates on paired augmented representations (B, D).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i, z_j: Feature projections of shape (B, D) from two differently augmented views of batch.
        Returns:
            Scalar NT-Xent loss.
        """
        B, D = z_i.shape
        if B < 1:
            return torch.tensor(0.0, device=z_i.device)
            
        z = torch.cat([z_i, z_j], dim=0) # (2B, D)
        z = F.normalize(z, p=2, dim=-1)
        
        # Similarity matrix (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Remove diagonal self-similarity
        mask_diag = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask_diag, -1e9)
        
        # Targets: pos pair for i is i + B (and vice versa)
        targets = torch.arange(2 * B, device=z.device)
        targets = (targets + B) % (2 * B)
        
        loss = F.cross_entropy(sim_matrix, targets)
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for Contrastive Predictive Coding (CPC) baseline.
    Predicts future representation state from past contextual state and discriminates against negative batch distractors.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, context_proj: torch.Tensor, future_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_proj: Projected context representations c_t of shape (B, D).
            future_target: Future actual representations z_{t+k} of shape (B, D).
        Returns:
            Scalar InfoNCE loss value.
        """
        B, D = context_proj.shape
        c = F.normalize(context_proj, p=2, dim=-1)
        z = F.normalize(future_target, p=2, dim=-1)
        
        # Dot product similarity between each predicted context and all future targets in batch
        sim_matrix = torch.matmul(c, z.T) / self.temperature  # shape: (B, B)
        
        # Positive pairs lie along the diagonal (i == j)
        targets = torch.arange(B, device=c.device)
        loss = F.cross_entropy(sim_matrix, targets)
        return loss
