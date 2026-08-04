"""
Kinematic equations for virtual IMU signal synthesis from 3D trajectory data.
Implements Equations (1) and (2) from SSSL-HAR (Li et al., IJCB 2025).
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional


def compute_smoothed_acceleration(
    positions: Union[np.ndarray, torch.Tensor],
    dt: float = 1.0 / 60.0,
    n: int = 4,
    pad_mode: str = "edge"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes smoothed acceleration from global 3D position trajectories using Eq. (1) in SSSL-HAR:
    
        a_i(t) = (x_i(t - n) + x_i(t + n) - 2 * x_i(t)) / (n * dt)^2
        
    Args:
        positions: Array or Tensor of shape (..., T, 3) representing 3D coordinates over time T.
        dt: Sampling interval in seconds (default: 1/60s for 60Hz).
        n: Smoothing intensity parameter (frame interval). Higher n reduces high-frequency MoCap noise.
        pad_mode: Padding strategy ('edge', 'zeros', or 'reflect') to preserve sequence duration T.
        
    Returns:
        Accelerations in global coordinate frame of same shape as positions (..., T, 3).
    """
    is_tensor = torch.is_tensor(positions)
    pos_np = positions.cpu().numpy() if is_tensor else np.asarray(positions)
    
    if pos_np.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3 (X, Y, Z coordinates), got {pos_np.shape[-1]}")
        
    T = pos_np.shape[-2]
    if T < 2 * n + 1:
        # If sequence is too short for large n, fallback to smaller n or 1
        n = max(1, (T - 1) // 2)
        if n < 1:
            return np.zeros_like(pos_np) if not is_tensor else torch.zeros_like(positions)

    # Pad along time axis (-2) to handle boundary frames
    pad_width = [(0, 0)] * pos_np.ndim
    pad_width[-2] = (n, n)
    padded_pos = np.pad(pos_np, pad_width, mode=pad_mode)
    
    # Extract shifted slices: x(t-n), x(t), x(t+n)
    x_prev = padded_pos[..., 0 : T, :]
    x_curr = padded_pos[..., n : T + n, :]
    x_next = padded_pos[..., 2 * n : T + 2 * n, :]
    
    # Equation (1) calculation
    denom = (n * dt) ** 2
    acc = (x_prev + x_next - 2.0 * x_curr) / denom
    
    if is_tensor:
        return torch.from_numpy(acc).to(positions.device, dtype=positions.dtype)
    return acc


def transform_to_local_frame(
    global_acceleration: Union[np.ndarray, torch.Tensor],
    orientations: Union[np.ndarray, torch.Tensor],
    gravity: Optional[Union[np.ndarray, torch.Tensor, Tuple[float, float, float]]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transforms global acceleration into local sensor frame readings accounting for gravity using Eq. (2):
    
        a_i^{loc}(t) = R_i(t) * (a_i(t) - g)
        
    Args:
        global_acceleration: Array or Tensor of shape (..., T, 3).
        orientations: Rotation matrices R_i(t) representing transformation from world to local sensor frame,
                      shape (..., T, 3, 3).
        gravity: Gravity vector in global coordinate frame. Defaults to [0.0, 0.0, -9.81] (Z-up world).
        
    Returns:
        Local IMU accelerometer readings of shape (..., T, 3) in m/s^2.
    """
    is_tensor = torch.is_tensor(global_acceleration)
    acc_np = global_acceleration.cpu().numpy() if is_tensor else np.asarray(global_acceleration)
    rot_np = orientations.cpu().numpy() if torch.is_tensor(orientations) else np.asarray(orientations)
    
    if gravity is None:
        g = np.array([0.0, 0.0, -9.81], dtype=acc_np.dtype)
    else:
        g = gravity.cpu().numpy() if torch.is_tensor(gravity) else np.asarray(gravity, dtype=acc_np.dtype)
        
    # Subtract gravity in global coordinate frame: (a_i(t) - g)
    # Note: an accelerometer at rest in gravity g=[-9.81] measures specific force +9.81 upward.
    kinematic_force = acc_np - g  # shape: (..., T, 3)
    
    # Rotate into local coordinate frame: R_i(t) * (a_i(t) - g)
    # Using matrix vector multiplication along last two dimensions:
    # rot_np has shape (..., T, 3, 3), kinematic_force has shape (..., T, 3, 1)
    kinematic_force_vec = kinematic_force[..., np.newaxis]
    local_acc = np.matmul(rot_np, kinematic_force_vec)[..., 0]
    
    if is_tensor:
        return torch.from_numpy(local_acc).to(global_acceleration.device, dtype=global_acceleration.dtype)
    return local_acc


def compute_angular_velocity(
    orientations: Union[np.ndarray, torch.Tensor],
    dt: float = 1.0 / 60.0,
    n: int = 2,
    pad_mode: str = "edge"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes local angular velocity \omega_i(t) from rotation matrices sequence R_i(t) by finite differentiation.
    
    Args:
        orientations: Rotation matrices of shape (..., T, 3, 3) from global to local sensor frame.
        dt: Sampling interval in seconds (default: 1/60s).
        n: Smoothing difference step across time frame.
        pad_mode: Padding strategy along temporal axis.
        
    Returns:
        Local angular velocity array/tensor of shape (..., T, 3) in rad/s.
    """
    is_tensor = torch.is_tensor(orientations)
    rot_np = orientations.cpu().numpy() if is_tensor else np.asarray(orientations)
    
    T = rot_np.shape[-3]
    if T < 2 * n + 1:
        n = max(1, (T - 1) // 2)
        if n < 1:
            return np.zeros_like(rot_np[..., 0]) if not is_tensor else torch.zeros_like(orientations[..., 0])
            
    # Pad along time axis (-3)
    pad_width = [(0, 0)] * rot_np.ndim
    pad_width[-3] = (n, n)
    padded_rot = np.pad(rot_np, pad_width, mode=pad_mode)
    
    # Extract R(t-n) and R(t+n)
    R_prev = padded_rot[..., 0 : T, :, :]
    R_next = padded_rot[..., 2 * n : T + 2 * n, :, :]
    
    # Relative rotation matrix Delta R = R(t+n) * R(t-n)^T
    # For infinitesimal rotation Delta R \approx I + [w * 2*n*dt]_\times
    R_prev_T = np.swapaxes(R_prev, -1, -2)
    R_rel = np.matmul(R_next, R_prev_T)
    
    # Extract skew-symmetric terms to find rotational velocity vector in global frame
    w_x = (R_rel[..., 2, 1] - R_rel[..., 1, 2]) / (4.0 * n * dt)
    w_y = (R_rel[..., 0, 2] - R_rel[..., 2, 0]) / (4.0 * n * dt)
    w_z = (R_rel[..., 1, 0] - R_rel[..., 0, 1]) / (4.0 * n * dt)
    w_global = np.stack([w_x, w_y, w_z], axis=-1)
    
    # Rotate angular velocity vector into local sensor coordinates at time t
    R_curr = rot_np
    w_local = np.matmul(R_curr, w_global[..., np.newaxis])[..., 0]
    
    if is_tensor:
        return torch.from_numpy(w_local).to(orientations.device, dtype=orientations.dtype)
    return w_local
