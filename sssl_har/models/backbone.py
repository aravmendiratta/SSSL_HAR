"""
Multi-View Encoders and Aggregator architecture as detailed in Section 4.2.2 and Figure 2.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class SingleViewEncoder(nn.Module):
    """
    Three-layer 1D Convolutional Network encoder for an individual sensor view (e.g., 3-axis acc or gyro).
    Progressively expands channels (16 -> 32 -> 64) and shrinks kernel sizes (24 -> 16 -> 8),
    with Layer Normalization and PReLU activation, followed by Global Average Pooling.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 64):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=24, padding=11)
        self.norm1 = nn.LayerNorm(16)
        self.act1 = nn.PReLU(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, padding=7)
        self.norm2 = nn.LayerNorm(32)
        self.act2 = nn.PReLU(32)
        
        self.conv3 = nn.Conv1d(32, out_dim, kernel_size=8, padding=3)
        self.norm3 = nn.LayerNorm(out_dim)
        self.act3 = nn.PReLU(out_dim)
        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _apply_block(self, x: torch.Tensor, conv: nn.Module, norm: nn.LayerNorm, act: nn.Module) -> torch.Tensor:
        x = conv(x)
        # Permute for LayerNorm over channel dimension: (B, C, T) -> (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = norm(x)
        x = x.permute(0, 2, 1)
        x = act(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sensor input tensor of shape (B, C=3, T).
        Returns:
            View embedding of shape (B, 64).
        """
        x = self._apply_block(x, self.conv1, self.norm1, self.act1)
        x = self._apply_block(x, self.conv2, self.norm2, self.act2)
        x = self._apply_block(x, self.conv3, self.norm3, self.act3)
        
        x = self.pool(x) # (B, 64, 1)
        return x.squeeze(-1) # (B, 64)


class MultiViewEncoder(nn.Module):
    """
    Collection of independent SingleViewEncoders (E_1, E_2, ..., E_N) to encode each sensor position/type
    independently into 64-dimensional embeddings.
    """

    def __init__(self, num_views: int = 6, in_channels_per_view: int = 3, embedding_dim: int = 64):
        """
        Args:
            num_views: Number of distinct sensor views N (e.g. 6 for 3ACC+3GYRO, 3 for 3ACC).
            in_channels_per_view: Typically 3 axes (X, Y, Z).
            embedding_dim: Output dimension of each view (default: 64).
        """
        super().__init__()
        self.num_views = num_views
        self.embedding_dim = embedding_dim
        self.encoders = nn.ModuleList([
            SingleViewEncoder(in_channels=in_channels_per_view, out_dim=embedding_dim)
            for _ in range(num_views)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Multi-view tensor of shape (B, N_views, C=3, T).
        Returns:
            Multi-view embedding matrix of shape (B, N_views, 64).
        """
        B, N, C, T = x.shape
        if N != self.num_views:
            raise ValueError(f"Expected {self.num_views} sensor views, but got input with {N} views.")
            
        view_embeddings = []
        for i in range(self.num_views):
            z_v = self.encoders[i](x[:, i, :, :])  # (B, 64)
            view_embeddings.append(z_v)
            
        return torch.stack(view_embeddings, dim=1)  # (B, N, 64)


class Aggregator(nn.Module):
    """
    Two-layer fully-connected aggregator to fuse multi-view embeddings (B, N_views, 64)
    into a unified aggregated representation (B, 256) as shown in Figure 2.
    """

    def __init__(self, num_views: int = 6, view_dim: int = 64, agg_dim: int = 256):
        super().__init__()
        in_dim = num_views * view_dim
        self.fc1 = nn.Linear(in_dim, agg_dim)
        self.norm1 = nn.LayerNorm(agg_dim)
        self.act1 = nn.PReLU(agg_dim)
        
        self.fc2 = nn.Linear(agg_dim, agg_dim)
        self.norm2 = nn.LayerNorm(agg_dim)

    def forward(self, view_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            view_embeddings: Multi-view feature matrix of shape (B, N, 64).
            mask: Optional spatial masking tensor of shape (B, N) or (B, N, 1) for CroSSL pretraining.
                  When masked, masked sensor view embeddings are set to 0.
        Returns:
            Aggregated representation of shape (B, 256).
        """
        B, N, D = view_embeddings.shape
        x = view_embeddings
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1) # (B, N, 1)
            x = x * mask
            
        x_flat = x.reshape(B, N * D)
        out = self.fc1(x_flat)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return out


class SSSLBackbone(nn.Module):
    """
    Complete feature extraction backbone combining MultiViewEncoder and Aggregator.
    Serves as the core representation module across both SSL pretraining and supervised fine-tuning.
    """

    def __init__(self, num_views: int = 6, in_channels_per_view: int = 3, view_dim: int = 64, agg_dim: int = 256):
        super().__init__()
        self.encoder = MultiViewEncoder(num_views=num_views, in_channels_per_view=in_channels_per_view, embedding_dim=view_dim)
        self.aggregator = Aggregator(num_views=num_views, view_dim=view_dim, agg_dim=agg_dim)

    def forward(self, x: torch.Tensor, return_views: bool = False, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Multi-view IMU tensor of shape (B, N, C, T).
            return_views: If True, returns tuple of (aggregated_embed, view_embeddings).
            mask: Optional view mask for CroSSL training.
        Returns:
            aggregated_embed (B, 256) or (aggregated_embed, view_embeddings).
        """
        view_embeds = self.encoder(x) # (B, N, 64)
        agg_embed = self.aggregator(view_embeds, mask=mask) # (B, 256)
        if return_views:
            return agg_embed, view_embeds
        return agg_embed
