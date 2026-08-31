import numpy as np
import torch

def generate_darcy_sample(resolution=64, seed=None, angle_deg=0.0):
    """Generates Darcy flow sample with exact analytical gradients."""
    if seed is not None:
        np.random.seed(seed)
    nx = ny = resolution
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.flatten(), Y.flatten()], axis=-1)

    theta = angle_deg * np.pi / 180.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    if angle_deg != 0:
        coords_rot = coords @ R.T
        X_r = coords_rot[:, 0].reshape(nx, ny)
        Y_r = coords_rot[:, 1].reshape(nx, ny)
    else:
        coords_rot = coords
        X_r = X
        Y_r = Y

    # Permeability field (k)
    k = 1.0 + 0.3 * np.sin(2*np.pi*X_r + 1.0) * np.cos(3*np.pi*Y_r - 0.5)
    k += 0.2 * np.sin(4*np.pi*X_r*Y_r)
    k = np.clip(k, 0.5, 2.0)

    # Pressure field (p) - Ground Truth
    p = np.sin(2*np.pi*X_r) * np.cos(2*np.pi*Y_r) + 0.5*np.sin(5*np.pi*X_r+1.3)*np.cos(4*np.pi*Y_r-0.7)

    # Exact Analytical Gradients for k
    dk_dX = (0.3 * (2*np.pi) * np.cos(2*np.pi*X_r + 1.0) * np.cos(3*np.pi*Y_r - 0.5) +
             0.2 * (4*np.pi*Y_r) * np.cos(4*np.pi*X_r*Y_r))
    dk_dY = (-0.3 * (3*np.pi) * np.sin(2*np.pi*X_r + 1.0) * np.sin(3*np.pi*Y_r - 0.5) +
             0.2 * (4*np.pi*X_r) * np.cos(4*np.pi*X_r*Y_r))

    grad_k_vec = np.stack([dk_dX.flatten(), dk_dY.flatten()], axis=-1)
    grad_k_mag = np.sqrt((grad_k_vec**2).sum(-1)).flatten()

    return coords_rot, k.flatten(), grad_k_mag, grad_k_vec, p.flatten()