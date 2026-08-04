"""
Engine module managing pretraining, fine-tuning, and evaluation loops for SSSL-HAR.
"""

from .trainer import SSSLTrainer, train_and_evaluate_experiment

__all__ = [
    "SSSLTrainer",
    "train_and_evaluate_experiment",
]
