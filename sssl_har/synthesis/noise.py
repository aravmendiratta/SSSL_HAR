"""
Physical IMU noise injection modeling based on the PNP (Physical Non-inertial Poser) framework.
Implements random walk sensor bias and additive Gaussian white noise as described in Section 4.3.1.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class PhysicalNoiseConfig:
    """
    Configuration for realistic IMU physical noise modeling.
    
    Attributes:
        acc_noise_std: Standard deviation of accelerometer additive white noise (m/s^2).
        acc_bias_std: Standard deviation of accelerometer random walk drift rate (m/s^2 / sqrt(s)).
        gyro_noise_std: Standard deviation of gyroscope additive white noise (rad/s).
        gyro_bias_std: Standard deviation of gyroscope random walk drift rate (rad/s / sqrt(s)).
        sample_rate: Sensor sampling rate in Hz (default: 60 Hz).
        seed: Optional random seed for deterministic simulation.
    """
    acc_noise_std: float = 0.05       # typical MEMS accelerometer noise density
    acc_bias_std: float = 0.005       # bias instability / random walk
    gyro_noise_std: float = 0.005     # typical MEMS gyroscope noise density (rad/s)
    gyro_bias_std: float = 0.0005     # bias instability / random walk (rad/s)
    sample_rate: float = 60.0
    seed: Optional[int] = None


def inject_sensor_noise(
    signals: Union[np.ndarray, torch.Tensor],
    sensor_type: str = "acc",
    config: Optional[PhysicalNoiseConfig] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Injects realistic physical measurement noise into synthesized IMU signals:
    Total Measurement Noise = Sensor Bias (Random Walk) + Additive Gaussian White Noise.
    
    Args:
        signals: Array or Tensor of shape (..., T, 3) representing accelerometer or gyroscope readings over time.
        sensor_type: Type of sensor ('acc' for accelerometer, 'gyro' for gyroscope).
        config: Noise parameter configuration (PhysicalNoiseConfig). Defaults to typical MEMS sensors if None.
        
    Returns:
        Noisy sensor readings of identical shape and type as input signals.
    """
    if config is None:
        config = PhysicalNoiseConfig()
        
    is_tensor = torch.is_tensor(signals)
    sig_np = signals.cpu().numpy().astype(np.float64) if is_tensor else np.asarray(signals, dtype=np.float64)
    
    rng = np.random.default_rng(config.seed)
    
    T = sig_np.shape[-2]
    dt = 1.0 / config.sample_rate
    
    if sensor_type.lower().startswith("acc"):
        noise_std = config.acc_noise_std
        bias_std = config.acc_bias_std
    elif sensor_type.lower().startswith("gyro") or sensor_type.lower().startswith("gyr"):
        noise_std = config.gyro_noise_std
        bias_std = config.gyro_bias_std
    else:
        raise ValueError(f"Unknown sensor_type '{sensor_type}', expected 'acc' or 'gyro'.")
        
    # 1. Additive Gaussian White Noise
    white_noise = rng.normal(loc=0.0, scale=noise_std, size=sig_np.shape)
    
    # 2. Random Walk Sensor Bias
    # Bias changes by a Gaussian step at each sample scaled by sqrt(dt)
    bias_steps = rng.normal(loc=0.0, scale=bias_std * np.sqrt(dt), size=sig_np.shape)
    random_walk_bias = np.cumsum(bias_steps, axis=-2)
    
    # Optional initial static bias offset per sequence
    init_bias = rng.normal(loc=0.0, scale=bias_std * 5.0, size=(*sig_np.shape[:-2], 1, sig_np.shape[-1]))
    total_bias = random_walk_bias + init_bias
    
    noisy_signals = sig_np + white_noise + total_bias
    
    if is_tensor:
        return torch.from_numpy(noisy_signals).to(signals.device, dtype=signals.dtype)
    return noisy_signals.astype(np.asarray(signals).dtype)
