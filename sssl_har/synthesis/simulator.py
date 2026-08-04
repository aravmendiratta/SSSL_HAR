"""
Synthetic human movement and biomechanical joint trajectory simulator.
Generates multi-joint 3D positions and rotations to emulate AMASS/SMPL kinematic sequences
and produces synchronized multi-view IMU datasets for pre-training and experimentation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
from .kinematics import compute_smoothed_acceleration, transform_to_local_frame, compute_angular_velocity
from .noise import inject_sensor_noise, PhysicalNoiseConfig


class KinematicTrajectorySimulator:
    """
    Simulates multi-joint human kinematics across diverse physical activities (e.g., rhythmic running,
    repetitive gym exercises, rotational stretching, and stationary poses) to synthesize virtual IMU views.
    """

    def __init__(
        self,
        sample_rate: float = 60.0,
        joint_names: Optional[List[str]] = None,
        seed: Optional[int] = 42
    ):
        """
        Args:
            sample_rate: Simulated simulation frequency in Hz (e.g., 60Hz or 120Hz for AMASS).
            joint_names: List of anatomical tracking positions (default: ['wrist', 'chest', 'ankle']).
            seed: Random seed for reproducible generation.
        """
        self.sample_rate = sample_rate
        self.joint_names = joint_names or ["wrist", "chest", "ankle"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_joint_trajectory(
        self,
        duration_sec: float = 5.0,
        activity_type: str = "dynamic",
        joint_name: str = "wrist"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates continuous 3D coordinate trajectories and local frame rotation matrices for a joint.
        
        Args:
            duration_sec: Length of simulated motion sequence in seconds.
            activity_type: Type of motion ('dynamic', 'rhythmic', 'static', 'complex').
            joint_name: Target anatomical location influencing motion amplitude and frequency.
            
        Returns:
            positions: 3D position trajectories of shape (T, 3) in meters.
            orientations: 3D rotation matrices of shape (T, 3, 3).
        """
        T = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, T, endpoint=False)
        
        # Anatomical characteristic frequency and amplitude scaling
        if joint_name in ["wrist", "upper_arm", "arm"]:
            base_freq = 1.5 + self.rng.uniform(-0.3, 0.5)  # fast arm swings
            amp = np.array([0.4, 0.4, 0.3])
        elif joint_name in ["ankle", "calf", "thigh", "leg"]:
            base_freq = 2.0 + self.rng.uniform(-0.2, 0.4)  # locomotion stepping
            amp = np.array([0.5, 0.3, 0.25])
        else:
            # chest, pelvis, torso
            base_freq = 1.0 + self.rng.uniform(-0.1, 0.2)  # stable torso bounce
            amp = np.array([0.15, 0.15, 0.1])
            
        if activity_type == "static":
            # Very small breathing tremor or slight shifting
            amp *= 0.05
            base_freq = 0.3
        elif activity_type == "rhythmic":
            # Consistent periodic motion (e.g. running, cycling)
            pass
        elif activity_type == "complex":
            # Multi-harmonic fitness exercise with abrupt direction reversals
            amp *= 1.3
            
        # Generate 3D harmonic trajectory + mild perturbation
        phase = self.rng.uniform(0, 2 * np.pi, size=3)
        pos_x = amp[0] * np.sin(2 * np.pi * base_freq * t + phase[0]) + 0.02 * np.sin(2 * np.pi * 3.7 * t)
        pos_y = amp[1] * np.cos(2 * np.pi * (base_freq * 0.5) * t + phase[1])
        pos_z = 1.0 + amp[2] * np.sin(2 * np.pi * (base_freq * 1.5) * t + phase[2])  # elevated above ground
        
        positions = np.stack([pos_x, pos_y, pos_z], axis=-1)
        
        # Generate associated orientations R_i(t) via Euler angles (roll, pitch, yaw over time)
        roll = 0.5 * np.sin(2 * np.pi * base_freq * t + phase[0])
        pitch = 0.4 * np.cos(2 * np.pi * (base_freq * 0.8) * t + phase[1])
        yaw = 0.3 * np.sin(2 * np.pi * (base_freq * 0.3) * t + phase[2])
        
        orientations = np.zeros((T, 3, 3), dtype=np.float64)
        for i in range(T):
            r, p, y = roll[i], pitch[i], yaw[i]
            Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
            Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
            Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
            orientations[i] = Rz @ Ry @ Rx
            
        return positions, orientations

    def synthesize_multi_view_imu(
        self,
        duration_sec: float = 5.0,
        activity_type: str = "dynamic",
        smoothing_n: int = 4,
        add_noise: bool = True,
        noise_config: Optional[PhysicalNoiseConfig] = None
    ) -> Dict[str, np.ndarray]:
        """
        Synthesizes synchrononous multi-view accelerometer and gyroscope signals across all joints.
        
        Returns:
            Dictionary mapping sensor view identifiers (e.g. 'wrist_acc', 'wrist_gyro') to arrays of shape (T, 3).
        """
        dt = 1.0 / self.sample_rate
        if noise_config is None and add_noise:
            noise_config = PhysicalNoiseConfig(sample_rate=self.sample_rate)
            
        imu_data = {}
        for joint in self.joint_names:
            pos, rot = self.generate_joint_trajectory(duration_sec, activity_type, joint)
            
            # 1. Smoothed differentiation to get global acceleration (Eq. 1)
            glob_acc = compute_smoothed_acceleration(pos, dt=dt, n=smoothing_n)
            
            # 2. Transform into local sensor coordinate frame accounting for gravity (Eq. 2)
            local_acc = transform_to_local_frame(glob_acc, rot)
            
            # 3. Compute local angular velocity
            local_gyro = compute_angular_velocity(rot, dt=dt, n=max(1, smoothing_n // 2))
            
            # 4. Inject physical noise
            if add_noise:
                local_acc = inject_sensor_noise(local_acc, sensor_type="acc", config=noise_config)
                local_gyro = inject_sensor_noise(local_gyro, sensor_type="gyro", config=noise_config)
                
            imu_data[f"{joint}_acc"] = local_acc
            imu_data[f"{joint}_gyro"] = local_gyro
            imu_data[f"{joint}_pos_3d"] = pos  # retained for visual lab demonstration
            
        return imu_data

    def generate_dataset_batch(
        self,
        num_samples: int = 100,
        window_size: int = 256,
        smoothing_n: int = 4,
        include_gyro: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generates a batch of synchronized multi-view IMU tensor sequences for self-supervised training.
        
        Args:
            num_samples: Number of activity samples / sequences to generate.
            window_size: Sequence duration T per window (e.g., 256 for PAMAP2, 512 for custom fitness).
            smoothing_n: Smoothing interval for acceleration calculation.
            include_gyro: Whether to include gyroscope views or only accelerometers (3ACC+3GYRO vs 3ACC).
            
        Returns:
            multi_view_tensor: PyTorch Tensor of shape (B, N_views, C=3, T).
            view_names: Ordered list of sensor view labels.
        """
        duration_sec = window_size / self.sample_rate
        activity_modes = ["dynamic", "rhythmic", "static", "complex"]
        
        batch_tensors = []
        view_names_out = None
        
        for _ in range(num_samples):
            act = self.rng.choice(activity_modes)
            sample_dict = self.synthesize_multi_view_imu(duration_sec, act, smoothing_n=smoothing_n)
            
            views_list = []
            current_view_names = []
            for k, v in sample_dict.items():
                if k.endswith("_pos_3d"):
                    continue
                if not include_gyro and k.endswith("_gyro"):
                    continue
                # v has shape (T, 3), transpose to (3, T) for CNN input (C, T)
                views_list.append(v.T[:, :window_size])
                current_view_names.append(k)
                
            if view_names_out is None:
                view_names_out = current_view_names
                
            batch_tensors.append(np.stack(views_list, axis=0))  # shape: (N_views, 3, T)
            
        arr = np.stack(batch_tensors, axis=0).astype(np.float32)  # shape: (B, N_views, 3, T)
        return torch.from_numpy(arr), view_names_out
