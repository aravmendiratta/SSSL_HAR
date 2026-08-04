"""
Unit tests for MVCL (COCOA, CroSSL/VICReg) and baseline loss functions.
"""

import unittest
import torch
from sssl_har.losses import COCOALoss, VICRegLoss, NTXentLoss, InfoNCELoss


class TestSSLLosses(unittest.TestCase):

    def test_cocoa_loss_gradients(self):
        loss_fn = COCOALoss(temperature=0.1, lambda_coeff=1.0)
        # Shape: (B, N_views, D_embed)
        view_embeddings = torch.randn(16, 6, 64, requires_grad=True)
        loss = loss_fn(view_embeddings)
        self.assertIsInstance(loss.item(), float)
        loss.backward()
        self.assertIsNotNone(view_embeddings.grad)
        self.assertFalse(torch.isnan(view_embeddings.grad).any())

    def test_vicreg_crossl_loss_gradients(self):
        loss_fn = VICRegLoss()
        z1 = torch.randn(32, 256, requires_grad=True)
        z2 = torch.randn(32, 256, requires_grad=True)
        loss = loss_fn(z1, z2)
        loss.backward()
        self.assertIsNotNone(z1.grad)
        self.assertIsNotNone(z2.grad)
        self.assertFalse(torch.isnan(loss))

    def test_ntxent_loss_gradients(self):
        loss_fn = NTXentLoss(temperature=0.5)
        h1 = torch.randn(24, 256, requires_grad=True)
        h2 = torch.randn(24, 256, requires_grad=True)
        loss = loss_fn(h1, h2)
        loss.backward()
        self.assertIsNotNone(h1.grad)

    def test_infonce_cpc_loss_gradients(self):
        loss_fn = InfoNCELoss()
        c_proj = torch.randn(16, 256, requires_grad=True)
        z_target = torch.randn(16, 256, requires_grad=True)
        loss = loss_fn(c_proj, z_target)
        loss.backward()
        self.assertIsNotNone(c_proj.grad)


if __name__ == "__main__":
    unittest.main()
