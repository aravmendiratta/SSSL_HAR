"""
Unit tests for deep learning architecture modules (MultiViewEncoder, Aggregator, HARClassifier).
"""

import unittest
import torch
from sssl_har.models import MultiViewEncoder, Aggregator, SSSLBackbone, HARClassifier
from sssl_har.methods import get_ssl_method_trainer


class TestModelsAndMethods(unittest.TestCase):

    def test_multi_view_encoder_shapes(self):
        encoder = MultiViewEncoder(num_views=6, in_channels_per_view=3, embedding_dim=64)
        # Batch size=8, Views=6, Channels=3, Length=256
        x = torch.randn(8, 6, 3, 256)
        out = encoder(x)
        self.assertEqual(out.shape, (8, 6, 64))

    def test_aggregator_with_mask(self):
        agg = Aggregator(num_views=6, view_dim=64, agg_dim=256)
        view_embeds = torch.randn(4, 6, 64)
        # Create Bernoulli spatial mask
        mask = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]] * 4)
        out = agg(view_embeds, mask=mask)
        self.assertEqual(out.shape, (4, 256))

    def test_har_classifier_forward(self):
        backbone = SSSLBackbone(num_views=3, agg_dim=256)
        classifier = HARClassifier(num_classes=11, backbone=backbone)
        x = torch.randn(5, 3, 3, 256)
        logits, feats = classifier(x, return_features=True)
        self.assertEqual(logits.shape, (5, 11))
        self.assertEqual(feats.shape, (5, 256))

    def test_ssl_method_trainers(self):
        backbone = SSSLBackbone(num_views=6)
        x = torch.randn(10, 6, 3, 256)
        
        for method in ["CroSSL", "COCOA", "SimCLR", "CPC"]:
            trainer = get_ssl_method_trainer(method, backbone)
            loss = trainer.compute_loss(x)
            self.assertIsInstance(loss.item(), float)
            self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
