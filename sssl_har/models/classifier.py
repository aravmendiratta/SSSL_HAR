"""
HAR classification head for supervised fine-tuning and evaluation.
"""

import torch
import torch.nn as nn
from .backbone import SSSLBackbone


class HARClassifier(nn.Module):
    """
    Human Activity Recognition classifier integrating a pretrained or random SSSLBackbone
    with an activity classification linear head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: SSSLBackbone,
        dropout_prob: float = 0.2
    ):
        """
        Args:
            num_classes: Number of target activity classes (e.g., 11 for PAMAP2, 25/43 for Custom Fitness).
            backbone: An instance of SSSLBackbone.
            dropout_prob: Dropout probability before final linear layer to prevent overfitting on minimal fine-tuning data.
        """
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_prob)
        agg_dim = backbone.aggregator.fc2.out_features
        self.fc_head = nn.Linear(agg_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input sensor tensor of shape (B, N_views, C=3, T).
            return_features: If True, returns (logits, aggregated_embedding).
        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.fc_head(self.dropout(features))
        if return_features:
            return logits, features
        return logits

    def freeze_backbone(self, freeze_encoder: bool = True, freeze_aggregator: bool = False):
        """
        Optionally freezes feature extraction weights for probing or adaptation.
        In standard SSSL-HAR fine-tuning, both feature extractor and classification head are trained with minimal labeled data.
        """
        for param in self.backbone.encoder.parameters():
            param.requires_grad = not freeze_encoder
        for param in self.backbone.aggregator.parameters():
            param.requires_grad = not freeze_aggregator
