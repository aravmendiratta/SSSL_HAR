"""
Virtual IMU Synthesis module for generating physically interpretable sensor signals
from 3D kinematic tracking data.
"""

from .kinematics import compute_smoothed_acceleration, transform_to_local_frame, compute_angular_velocity
from .noise import inject_sensor_noise, PhysicalNoiseConfig

__all__ = [
    "compute_smoothed_acceleration",
    "transform_to_local_frame",
    "compute_angular_velocity",
    "inject_sensor_noise",
    "PhysicalNoiseConfig",
]
