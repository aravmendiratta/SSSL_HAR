"""
Two-Stage SSL Training Engine for SSSL-HAR.
Manages Stage 1 (Unsupervised Multi-View Pretraining on Real/Synthetic data) and Stage 2 (Supervised Fine-tuning).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, Any, List
from tqdm import tqdm
import numpy as np

from ..models.backbone import SSSLBackbone
from ..models.classifier import HARClassifier
from ..methods.ssl_methods import get_ssl_method_trainer, BaseSSLMethod
from ..synthesis.simulator import KinematicTrajectorySimulator
from ..utils.metrics import compute_har_metrics


class SSSLTrainer:
    """
    Orchestrates pretraining, fine-tuning, and feature representation extraction.
    """

    def __init__(
        self,
        num_classes: int,
        num_views: int = 6,
        view_dim: int = 64,
        agg_dim: int = 256,
        device: Optional[str] = None,
        lr: float = 1e-4
    ):
        """
        Args:
            num_classes: Number of activity classification labels for fine-tuning stage.
            num_views: Number of input sensor views N (default: 6 for 3ACC+3GYRO).
            lr: Learning rate (default: 1e-4 for both stages as in Section 4.2.2).
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = SSSLBackbone(num_views=num_views, view_dim=view_dim, agg_dim=agg_dim).to(self.device)
        self.classifier = HARClassifier(num_classes=num_classes, backbone=self.backbone).to(self.device)
        self.lr = lr

    def pretrain(
        self,
        method_name: str,
        dataloader: Optional[DataLoader] = None,
        simulator: Optional[KinematicTrajectorySimulator] = None,
        num_synthetic_samples: int = 400,
        window_size: int = 256,
        epochs: int = 10,
        verbose: bool = True
    ) -> List[float]:
        """
        Stage 1: Pretrains feature backbone using unsupervised contrastive learning (CroSSL, COCOA, SimCLR, or CPC).
        
        Args:
            method_name: SSL framework identifier (e.g. 'CroSSL', 'COCOA', 'SimCLR').
            dataloader: Unlabeled DataLoader for real data pretraining ('-real' variants).
            simulator: KinematicSimulator instance for synthetic pretraining ('-synth' variants).
            epochs: Pretraining duration (100 epochs in full benchmark).
        """
        if method_name.lower().startswith("supervised") or method_name.lower() == "none":
            if verbose:
                print("[Pretraining Skipped] Supervised baseline learns end-to-end directly on fine-tuning data.")
            return []

        ssl_trainer = get_ssl_method_trainer(method_name, self.backbone).to(self.device)
        optimizer = torch.optim.Adam(ssl_trainer.parameters(), lr=self.lr)
        
        # If no dataloader is supplied, generate continuous virtual IMU batches via simulator
        if dataloader is None:
            if simulator is None:
                simulator = KinematicTrajectorySimulator(sample_rate=60.0)
            synth_tensor, _ = simulator.generate_dataset_batch(
                num_samples=num_synthetic_samples,
                window_size=window_size,
                include_gyro=(self.backbone.encoder.num_views == 6)
            )
            dataset = TensorDataset(synth_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

        loss_history = []
        iterator = tqdm(range(1, epochs + 1), desc=f"Stage 1 Pretrain [{method_name}]", disable=not verbose)
        
        ssl_trainer.train()
        for epoch in iterator:
            epoch_losses = []
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                loss = ssl_trainer.compute_loss(x)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                
            avg_loss = float(np.mean(epoch_losses))
            loss_history.append(avg_loss)
            if verbose:
                iterator.set_postfix(Loss=f"{avg_loss:.4f}")
                
        return loss_history

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 15,
        verbose: bool = True
    ) -> List[float]:
        """
        Stage 2: Supervised fine-tuning on labeled activity target set.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        
        loss_history = []
        iterator = tqdm(range(1, epochs + 1), desc="Stage 2 Finetuning", disable=not verbose)
        
        for epoch in iterator:
            self.classifier.train()
            epoch_losses = []
            for batch_x, batch_y in train_loader:
                x, y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.classifier(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                
            avg_loss = float(np.mean(epoch_losses))
            loss_history.append(avg_loss)
            if verbose:
                iterator.set_postfix(Loss=f"{avg_loss:.4f}")
                
        return loss_history

    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluates HAR performance metrics across test dataset and extracts aggregated feature embeddings.
        
        Returns:
            metrics_dict: Acc, Recall, Prec, F1_M, F1_W percentages.
            embeddings: Extracted 256-D features for all test samples.
            predictions: Predicted class indices.
        """
        self.classifier.eval()
        all_preds = []
        all_targets = []
        all_embeds = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                x = batch_x.to(self.device)
                logits, feats = self.classifier(x, return_features=True)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())
                all_embeds.append(feats.cpu().numpy())
                
        metrics = compute_har_metrics(all_targets, all_preds)
        embeddings = np.concatenate(all_embeds, axis=0) if all_embeds else np.empty((0, 256))
        
        return metrics, embeddings, np.array(all_preds)


def train_and_evaluate_experiment(
    method_variant: str,
    dataset_name: str = "pamap2",
    sensor_config: str = "3ACC+3GYRO",
    num_classes: int = 11,
    pretrain_loader: Optional[DataLoader] = None,
    finetune_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    pretrain_epochs: int = 8,
    finetune_epochs: int = 12,
    verbose: bool = False
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Helper function to run an entire pretrain -> finetune -> evaluate experimental trial for a single method variant
    (e.g. 'CroSSL-synth', 'COCOA-real', 'Supervised baseline').
    """
    num_views = 6 if "GYRO" in sensor_config.upper() else 3
    window_size = 256 if "pamap2" in dataset_name.lower() else 512
    
    trainer = SSSLTrainer(num_classes=num_classes, num_views=num_views)
    
    parts = method_variant.split("-")
    method_base = parts[0]
    is_synth = len(parts) > 1 and parts[1].lower().startswith("synth")
    
    if method_base.lower().startswith("supervised"):
        trainer.pretrain("supervised", verbose=verbose)
    elif is_synth:
        simulator = KinematicTrajectorySimulator(sample_rate=60.0)
        trainer.pretrain(method_base, simulator=simulator, window_size=window_size, epochs=pretrain_epochs, verbose=verbose)
    else:
        trainer.pretrain(method_base, dataloader=pretrain_loader, epochs=pretrain_epochs, verbose=verbose)
        
    if finetune_loader is not None:
        trainer.finetune(train_loader=finetune_loader, epochs=finetune_epochs, verbose=verbose)
        
    metrics, embeds, preds = trainer.evaluate(test_loader)
    return metrics, embeds, preds
