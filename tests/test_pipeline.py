"""
Integration tests for dataset loaders, pretraining engines, and evaluation pipeline.
"""

import unittest
from sssl_har.data import get_pamap2_dataloaders, get_fitness_dataloaders
from sssl_har.engine import SSSLTrainer, train_and_evaluate_experiment


class TestPipelineIntegration(unittest.TestCase):

    def test_pamap2_dataloader_and_mocking(self):
        pre, fine, test = get_pamap2_dataloaders(batch_size=16, sensor_config="3ACC+3GYRO", num_train_samples=40, num_test_samples=20)
        batch_x, batch_y = next(iter(fine))
        self.assertEqual(batch_x.shape, (16, 6, 3, 256))
        self.assertEqual(batch_y.shape, (16,))

    def test_custom_fitness_dataloader(self):
        pre, fine, test = get_fitness_dataloaders(batch_size=8, num_classes=25, window_size=512, num_train_samples=30, num_test_samples=16)
        batch_x, batch_y = next(iter(test))
        self.assertEqual(batch_x.shape, (8, 6, 3, 512))

    def test_trainer_lifecycle(self):
        pre_L, fine_L, test_L = get_pamap2_dataloaders(batch_size=16, num_train_samples=32, num_test_samples=16)
        trainer = SSSLTrainer(num_classes=11, num_views=6)
        
        # Test Stage 1 Pretraining
        loss_hist = trainer.pretrain(method_name="CroSSL", dataloader=pre_L, epochs=1, verbose=False)
        self.assertEqual(len(loss_hist), 1)
        
        # Test Stage 2 Finetuning
        fine_hist = trainer.finetune(train_loader=fine_L, epochs=1, verbose=False)
        self.assertEqual(len(fine_hist), 1)
        
        # Test Evaluation
        metrics, embeds, preds = trainer.evaluate(test_L)
        self.assertIn("Acc", metrics)
        self.assertIn("F1_M", metrics)
        self.assertEqual(embeds.shape[1], 256)
        self.assertEqual(len(preds), len(test_L.dataset))


if __name__ == "__main__":
    unittest.main()
