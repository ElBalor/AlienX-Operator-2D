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

def collate_darcy_batch(batch_size, resolutions, device, rotation_prob=0.3):
    """Creates a batch with random resolutions and rotations."""
    max_N = max([res**2 for res in resolutions])
    
    batch_k = torch.zeros(batch_size, max_N, device=device)
    batch_mag = torch.zeros(batch_size, max_N, device=device)
    batch_vec = torch.zeros(batch_size, max_N, 2, device=device)
    batch_coords = torch.zeros(batch_size, max_N, 2, device=device)
    batch_p = torch.zeros(batch_size, max_N, device=device)
    batch_mask = torch.zeros(batch_size, max_N, device=device)
    
    knn_cache = {}

    for i in range(batch_size):
        res = np.random.choice(resolutions)
        angle = np.random.uniform(0, 360) if np.random.rand() < rotation_prob else 0.0
        
        coords, k, mag, vec, p = generate_darcy_sample(res, seed=None, angle_deg=angle)
        N = res * res
        
        batch_k[i, :N] = torch.tensor(k, dtype=torch.float32, device=device)
        batch_mag[i, :N] = torch.tensor(mag, dtype=torch.float32, device=device)
        batch_vec[i, :N] = torch.tensor(vec, dtype=torch.float32, device=device)
        batch_coords[i, :N] = torch.tensor(coords, dtype=torch.float32, device=device)
        batch_p[i, :N] = torch.tensor(p, dtype=torch.float32, device=device)
        batch_mask[i, :N] = 1.0
        
        if res not in knn_cache:
            knn_cache[res] = get_deterministic_grid_knn(res, device)

    return batch_k, batch_mag, batch_vec, batch_coords, batch_p, batch_mask, knn_cache