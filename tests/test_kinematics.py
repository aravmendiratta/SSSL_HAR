"""
Unit tests for kinematics synthesis equations (Eqs. 1 & 2) and physical noise modeling.
"""

import unittest
import numpy as np
import torch
from sssl_har.synthesis import (
    compute_smoothed_acceleration,
    transform_to_local_frame,
    compute_angular_velocity,
    inject_sensor_noise,
    PhysicalNoiseConfig
)


class TestKinematicsSynthesis(unittest.TestCase):

    def test_smoothed_acceleration_numpy(self):
        # Create a quadratic position sequence x(t) = 0.5 * a * t^2 -> second derivative should be constant a
        dt = 0.01
        T = 50
        t = np.linspace(0, (T-1)*dt, T)
        a_true = 4.0
        pos = np.zeros((T, 3), dtype=np.float64)
        pos[:, 0] = 0.5 * a_true * (t ** 2)
        
        acc = compute_smoothed_acceleration(pos, dt=dt, n=2, pad_mode="reflect")
        self.assertEqual(acc.shape, (T, 3))
        # Interior frames should closely recover constant acceleration a_true
        self.assertTrue(np.allclose(acc[5:-5, 0], a_true, atol=0.1))

    def test_smoothed_acceleration_tensor(self):
        pos_tensor = torch.randn(10, 256, 3)
        acc = compute_smoothed_acceleration(pos_tensor, dt=1.0/60.0, n=4)
        self.assertTrue(torch.is_tensor(acc))
        self.assertEqual(acc.shape, (10, 256, 3))
        self.assertEqual(acc.device, pos_tensor.device)

    def test_local_frame_transformation_gravity(self):
        # A stationary IMU (acceleration = 0) experiencing world gravity g = [0, 0, -9.81]
        acc_global = np.zeros((100, 3), dtype=np.float64)
        # Identity orientation matrices
        rot_matrices = np.tile(np.eye(3), (100, 1, 1))
        
        acc_local = transform_to_local_frame(acc_global, rot_matrices, gravity=(0.0, 0.0, -9.81))
        self.assertEqual(acc_local.shape, (100, 3))
        # Specific force along Z axis should equal +9.81 m/s^2
        self.assertTrue(np.allclose(acc_local[:, 2], 9.81, atol=1e-5))

    def test_angular_velocity_computation(self):
        rot_matrices = torch.eye(3).unsqueeze(0).repeat(64, 1, 1)
        w_local = compute_angular_velocity(rot_matrices, dt=1.0/60.0, n=2)
        self.assertEqual(w_local.shape, (64, 3))

    def test_physical_noise_injection(self):
        clean_sig = torch.zeros(32, 3)
        cfg = PhysicalNoiseConfig(sample_rate=60.0, seed=123)
        noisy_sig = inject_sensor_noise(clean_sig, sensor_type="acc", config=cfg)
        self.assertEqual(noisy_sig.shape, clean_sig.shape)
        # Verify noise actually altered readings
        self.assertFalse(torch.allclose(noisy_sig, clean_sig))


if __name__ == "__main__":
    unittest.main()
