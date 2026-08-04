"""
Multi-View Contrastive Learning (MVCL) loss functions:
1. COCOA Loss (Cross-Modal Contrastive Alignment loss) from Eqs (3-5).
2. VICReg Loss (Variance, Invariance, Covariance Regularization loss for CroSSL) from Eqs (6-9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class COCOALoss(nn.Module):
    """
    COCOA Loss operates directly on multi-view sensor embeddings (B, N_views, D=64).
    Treats different sensor views at the same timestep as hard positives (cross-view consistency L_C)
    and same sensor view across different timesteps as hard negatives (view discrimination L_D).
    """

    def __init__(self, temperature: float = 0.1, lambda_coeff: float = 1.0):
        super().__init__()
        self.r = temperature
        self.lmbda = lambda_coeff

    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_embeddings: Tensor of shape (B, N, D), where B is batch size (time sample windows),
                             N is number of sensor views, D is embedding dimension (64).
        Returns:
            Scalar COCOA loss.
        """
        B, N, D = view_embeddings.shape
        if N < 2:
            raise ValueError("COCOALoss requires at least 2 sensor views.")
            
        # Normalize feature vectors along D dimension for cosine similarity computation
        z = F.normalize(view_embeddings, p=2, dim=-1) # (B, N, D)
        
        # 1. Cross-View Consistency Loss (L_C^t) - Eq. (3)
        # For each time window in batch, compute similarity across different sensor views (v != w)
        # z_t: (N, D) -> similarity matrix S: (B, N, N)
        sim_views = torch.bmm(z, z.transpose(1, 2)) # (B, N, N)
        # Create mask to exclude diagonal (v == w)
        mask_diag_N = torch.eye(N, device=z.device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
        # Eq (3): exp((1 - S_vw) / r)
        exp_diff_views = torch.exp((1.0 - sim_views) / self.r)
        # Zero out diagonal
        exp_diff_views = exp_diff_views.masked_fill(mask_diag_N, 0.0)
        # Sum over views w != v, mean over B and N
        L_C = exp_diff_views.sum(dim=(1, 2)) / (N * (N - 1))
        L_C_mean = L_C.mean()
        
        # 2. View Discrimination Loss (L_D^v) - Eq. (4)
        # For each sensor view, compute similarity across different timesteps t != t' in batch
        # z transposed to (N, B, D) -> similarity matrix S: (N, B, B)
        z_trans = z.transpose(0, 1) # (N, B, D)
        sim_time = torch.bmm(z_trans, z_trans.transpose(1, 2)) # (N, B, B)
        
        if B > 1:
            mask_diag_B = torch.eye(B, device=z.device, dtype=torch.bool).unsqueeze(0).expand(N, B, B)
            # Eq (4): exp(S_v,v^(t, t') / r)
            exp_sim_time = torch.exp(sim_time / self.r)
            exp_sim_time = exp_sim_time.masked_fill(mask_diag_B, 0.0)
            L_D = exp_sim_time.sum(dim=(1, 2)) / (B * (B - 1))
            L_D_mean = L_D.mean()
        else:
            L_D_mean = torch.tensor(0.0, device=z.device)
            
        # Eq (5): Total COCOA loss
        total_loss = L_C_mean + self.lmbda * L_D_mean
        return total_loss


class VICRegLoss(nn.Module):
    """
    VICReg Loss (for CroSSL framework) operates on aggregated multi-view embeddings z1, z2 (B, 256)
    generated through distinct random spatial masking across sensor views.
    Implements Equations (6), (7), (8), (9).
    """

    def __init__(self, sim_coeff: float = 1.0, var_coeff: float = 25.0, cov_coeff: float = 25.0, gamma: float = 1.0, eps: float = 1e-4):
        """
        Args:
            sim_coeff (\nu): Invariance coefficient (default: 1.0 or 25.0).
            var_coeff (\lambda): Variance preservation coefficient (default: 25.0).
            cov_coeff (\mu): Covariance decorrelation coefficient (default: 25.0).
            gamma: Target variance threshold (default: 1.0).
            eps: Numerical stabilization epsilon for square root.
        """
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: Aggregated representation matrices of shape (B, 256) from two distinct masked view sets.
        Returns:
            Scalar VICReg loss value.
        """
        B, D = z1.shape
        
        # 1. Invariance Loss (L_inv) - Eq. (8)
        # Mean squared error between aggregated representations
        L_inv = F.mse_loss(z1, z2)
        
        # 2. Variance Loss (L_var) - Eq. (6)
        # Maintain standard deviation above threshold gamma along batch dimension for each feature
        std1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std2 = torch.sqrt(z2.var(dim=0) + self.eps)
        L_var = torch.mean(F.relu(self.gamma - std1)) + torch.mean(F.relu(self.gamma - std2))
        
        # 3. Covariance Loss (L_cov) - Eq. (7)
        # Decorrelate features by driving off-diagonal elements of sample covariance to zero
        z1_centered = z1 - z1.mean(dim=0, keepdim=True)
        z2_centered = z2 - z2.mean(dim=0, keepdim=True)
        
        cov1 = (z1_centered.T @ z1_centered) / (B - 1) if B > 1 else torch.zeros((D, D), device=z1.device)
        cov2 = (z2_centered.T @ z2_centered) / (B - 1) if B > 1 else torch.zeros((D, D), device=z2.device)
        
        L_cov = (self.off_diagonal(cov1).pow(2).sum() / D) + (self.off_diagonal(cov2).pow(2).sum() / D)
        
        # Eq (9): Weighted combination
        loss = self.sim_coeff * L_inv + self.var_coeff * L_var + self.cov_coeff * L_cov
        return loss
