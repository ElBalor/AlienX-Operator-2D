"""
AlienX (ISN) - 2D Release v2: GPU Darcy Flow Data Generator
============================================================
Pull-back rotation: fields are evaluated in reference coordinates, then the
gradients are pushed forward with the rotation matrix. The domain always
stays within [-1, 1]^2 - no diamond stretching at 45 degrees.

Analytical gradients (no finite differences) - the fix for the numerical
gradient artifacts that produced the 45-degree residual in v1.

Eric Yaka (Elbalor / The Digital Necromancer)
"""

import math
import torch


def generate_darcy_sample_gpu(resolution, device, angle_deg=0.0, seed=None):
    """
    Generate a Darcy flow sample on the GPU with pull-back rotation.

    Args:
        resolution: grid size (nx = ny)
        device:     torch device
        angle_deg:  rotation angle of the permeability field, degrees
        seed:       optional RNG seed for reproducibility

    Returns:
        coords:     [N, 2]   grid coordinates in [-1, 1]^2
        k:          [N]      permeability field
        grad_k_mag: [N]      magnitude of grad k (pushed forward)
        grad_k_vec: [N, 2]   gradient vector of k (pushed forward)
        p:          [N]      pressure field (ground truth)
    """
    if seed is not None:
        torch.manual_seed(seed)

    nx = ny = resolution
    x = torch.linspace(-1, 1, nx, device=device)
    y = torch.linspace(-1, 1, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    theta = angle_deg * math.pi / 180.0
    R_inv = torch.tensor([[math.cos(-theta), -math.sin(-theta)],
                          [math.sin(-theta),  math.cos(-theta)]], device=device)
    coords_ref = coords @ R_inv.T
    X_r = coords_ref[:, 0].reshape(nx, ny)
    Y_r = coords_ref[:, 1].reshape(nx, ny)

    k = 1.0 + 0.3 * torch.sin(2*math.pi*X_r + 1.0) * torch.cos(3*math.pi*Y_r - 0.5)
    k += 0.2 * torch.sin(4*math.pi*X_r*Y_r)
    k = torch.clamp(k, 0.5, 2.0)

    p = torch.sin(2*math.pi*X_r) * torch.cos(2*math.pi*Y_r) + \
        0.5*torch.sin(5*math.pi*X_r + 1.3)*torch.cos(4*math.pi*Y_r - 0.7)

    dk_dX = (0.3 * (2*math.pi) * torch.cos(2*math.pi*X_r + 1.0) * torch.cos(3*math.pi*Y_r - 0.5) +
             0.2 * (4*math.pi*Y_r) * torch.cos(4*math.pi*X_r*Y_r))
    dk_dY = (-0.3 * (3*math.pi) * torch.sin(2*math.pi*X_r + 1.0) * torch.sin(3*math.pi*Y_r - 0.5) +
             0.2 * (4*math.pi*X_r) * torch.cos(4*math.pi*X_r*Y_r))

    grad_ref = torch.stack([dk_dX.flatten(), dk_dY.flatten()], dim=-1)

    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]], device=device)
    grad_k_vec = grad_ref @ R.T
    grad_k_mag = torch.sqrt((grad_k_vec**2).sum(-1)).flatten()

    return coords, k.flatten(), grad_k_mag, grad_k_vec, p.flatten()
