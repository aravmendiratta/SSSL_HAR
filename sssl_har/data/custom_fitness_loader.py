"""
Custom Fitness Monitoring Dataset Loader and Simulator.
Supports 25-class (Custom-25) and 43-class (Custom-43) fitness exercises across 14 subjects (11:2:1 split),
with 512-sample window segmentation, sensor placement mismatch simulation, and sampling rate alignment analysis.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict


CUSTOM_43_ACTIVITIES = [
    "Jogging", "High Knees", "Frog Jump", "High Kick Touch", "Squat", "Walking Turn",
    "Sit-Up", "Butt Kick", "Bent-Knee Side Kick", "Quick Steps with Arm Swing", "Hurdle Step",
    "Side Leg Swing", "Jumping Jacks", "Seated Toe Raise", "Side Lunge", "Walking Lunge with Twist",
    "Abdominal and Back Stretch", "Standing Side Bend Stretch", "Prone T-Shoulder Raise",
    "Plank with Alternating Arm Reach", "Standing Head Turn", "Lunge Jump", "Box Step-Up with Leg Swap",
    "V-Up", "Straight Leg Lunge", "Biceps Curl", "Arm Circles", "Bodyweight Lat Pulldown",
    "Standing Bent Over Pick Up", "Heel-to-Glute Single Arm Reach Stretch", "Standing Shoulder Press",
    "Maximal Fist Forward Reach Stretch", "Speed Walking", "Standing Head-Hand Resistance Stretch",
    "Crossover Step", "Calf Raise", "Standing Push Press", "Hip External Rotation", "Stride Run",
    "Stationary Quick Steps", "Four-point support", "Horse Stance", "One Leg Jump"
]

CUSTOM_25_ACTIVITIES = CUSTOM_43_ACTIVITIES[:25]


class CustomFitnessDataset(Dataset):
    """
    Dataset representing the Custom Fitness monitoring scenario (sensors at upper arm, pelvis, calf).
    Generates multi-sensor exercises with simulated placement mismatch and sampling rate differences.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/custom_fitness",
        split_mode: str = "finetune",
        num_classes: int = 25,
        window_size: int = 512,
        sample_rate: float = 72.0,
        is_aligned_sampling_rate: bool = True,
        use_mock_if_missing: bool = True,
        num_mock_samples: int = 500,
        placement_mismatch_mode: str = "all_aligned",
        seed: int = 42
    ):
        """
        Args:
            num_classes: 25 for Custom-25 subset, 43 for full Custom-43 set.
            window_size: Segmentation duration T (default: 512 samples as per Section 4.1).
            sample_rate: Sampling frequency in Hz (e.g. aligned 72Hz vs unaligned raw 148Hz).
            placement_mismatch_mode: Controlled mismatch ('all_aligned', 'arm_to_wrist', 'pelvis_to_chest', 'calf_thigh_exchange').
        """
        super().__init__()
        self.split_mode = split_mode
        self.num_classes = num_classes
        self.window_size = window_size
        self.sample_rate = 72.0 if is_aligned_sampling_rate else 148.0
        self.placement_mismatch = placement_mismatch_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.activity_list = CUSTOM_25_ACTIVITIES if num_classes == 25 else CUSTOM_43_ACTIVITIES
        self.num_views = 6  # 3 locations (upper arm, pelvis, calf) x (3ACC + 3GYRO)
        
        if use_mock_if_missing:
            self.data, self.labels = self._generate_simulated_fitness(num_mock_samples)
        else:
            raise FileNotFoundError(f"Raw custom dataset not located at '{data_dir}'.")

    def _generate_simulated_fitness(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_list = []
        label_list = []
        
        count = n_samples
        if self.split_mode == "pretrain_real":
            count = int(n_samples * 1.5)
        elif self.split_mode == "test":
            count = max(50, n_samples // 3)
            
        # When sampling rate is high (148Hz) without alignment, temporal duration of captured motion per 512 window is shorter (~3.4s vs ~7.1s at 72Hz)
        duration_sec = self.window_size / self.sample_rate
        t = np.linspace(0, duration_sec, self.window_size, endpoint=False)
        
        for i in range(count):
            act_idx = i % len(self.activity_list)
            act_name = self.activity_list[act_idx]
            
            views_tensor = np.zeros((self.num_views, 3, self.window_size), dtype=np.float32)
            
            # Simulating upper arm (0), pelvis (1), calf (2)
            for loc_idx, loc_name in enumerate(["upper_arm", "pelvis", "calf"]):
                # Specific gym exercises generate distinct frequencies across limbs
                if any(w in act_name.lower() for w in ["jump", "jog", "run", "jack"]):
                    freq = 2.4 + self.rng.uniform(-0.2, 0.3)
                    amp_acc = np.array([5.5, 5.0, 6.0]) * (1.6 if loc_name == "calf" else 1.0)
                elif any(w in act_name.lower() for w in ["stretch", "stance", "support", "plank"]):
                    freq = 0.3 + self.rng.uniform(-0.05, 0.05)
                    amp_acc = np.array([0.3, 0.3, 0.2])
                elif any(w in act_name.lower() for w in ["curl", "press", "arm", "shoulder"]):
                    freq = 1.0 + self.rng.uniform(-0.1, 0.2)
                    amp_acc = np.array([3.5, 3.0, 2.0]) * (1.8 if loc_name == "upper_arm" else 0.4)
                else:
                    # Squat, Lunge, Step, etc.
                    freq = 1.2 + self.rng.uniform(-0.15, 0.2)
                    amp_acc = np.array([3.0, 4.0, 4.5]) * (1.4 if loc_name == "pelvis" else 1.0)
                    
                # Apply Placement Mismatch Transformations (Fig. 6)
                if self.placement_mismatch == "arm_to_wrist" and loc_name == "upper_arm":
                    amp_acc *= 1.35  # wrist swings through larger arc than upper arm
                elif self.placement_mismatch == "pelvis_to_chest" and loc_name == "pelvis":
                    amp_acc *= 0.85
                elif self.placement_mismatch == "calf_thigh_exchange" and loc_name == "calf":
                    amp_acc *= 0.70
                    
                phase_acc = self.rng.uniform(0, 2 * np.pi, size=3)
                acc = np.stack([
                    amp_acc[0] * np.sin(2 * np.pi * freq * t + phase_acc[0]),
                    amp_acc[1] * np.cos(2 * np.pi * freq * t + phase_acc[1]),
                    amp_acc[2] * np.sin(2 * np.pi * (freq * 1.3) * t + phase_acc[2]) + 9.81
                ], axis=0) + self.rng.normal(0, 0.12, (3, self.window_size))
                
                phase_gyro = self.rng.uniform(0, 2 * np.pi, size=3)
                gyro = np.stack([
                    (amp_acc[0] * 0.4) * np.cos(2 * np.pi * freq * t + phase_gyro[0]),
                    (amp_acc[1] * 0.4) * np.sin(2 * np.pi * freq * t + phase_gyro[1]),
                    (amp_acc[2] * 0.3) * np.cos(2 * np.pi * (freq * 0.7) * t + phase_gyro[2])
                ], axis=0) + self.rng.normal(0, 0.06, (3, self.window_size))
                
                views_tensor[loc_idx] = acc
                views_tensor[loc_idx + 3] = gyro
                
            data_list.append(views_tensor)
            label_list.append(act_idx)
            
        return torch.from_numpy(np.stack(data_list, axis=0).astype(np.float32)), torch.from_numpy(np.array(label_list, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def get_fitness_dataloaders(
    batch_size: int = 64,
    num_classes: int = 25,
    window_size: int = 512,
    is_aligned_sampling_rate: bool = True,
    num_train_samples: int = 350,
    num_test_samples: int = 100,
    placement_mismatch_mode: str = "all_aligned"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (pretrain_loader, finetune_loader, test_loader) for Custom Fitness dataset experiments."""
    pretrain_set = CustomFitnessDataset(split_mode="pretrain_real", num_classes=num_classes, window_size=window_size, is_aligned_sampling_rate=is_aligned_sampling_rate, num_mock_samples=num_train_samples, placement_mismatch_mode=placement_mismatch_mode, seed=201)
    finetune_set = CustomFitnessDataset(split_mode="finetune", num_classes=num_classes, window_size=window_size, is_aligned_sampling_rate=is_aligned_sampling_rate, num_mock_samples=num_train_samples, placement_mismatch_mode=placement_mismatch_mode, seed=202)
    test_set = CustomFitnessDataset(split_mode="test", num_classes=num_classes, window_size=window_size, is_aligned_sampling_rate=is_aligned_sampling_rate, num_mock_samples=num_test_samples, placement_mismatch_mode=placement_mismatch_mode, seed=203)
    
    pretrain_loader = DataLoader(pretrain_set, batch_size=batch_size, shuffle=True, drop_last=True)
    finetune_loader = DataLoader(finetune_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return pretrain_loader, finetune_loader, test_loader
