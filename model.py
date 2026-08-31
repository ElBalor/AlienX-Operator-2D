import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------------------------------------------------------
# Cached Deterministic Grid KNN
# ------------------------------------------------------------------------------
_KNN_CACHE = {}

def get_deterministic_grid_knn(grid_size, device, k=8, dilation=1):
    """Builds a deterministic 8‑neighbor grid stencil with reflection padding."""
    N = grid_size * grid_size
    idx = torch.arange(N, device=device).view(1, 1, grid_size, grid_size)
    padded = F.pad(idx.float(), (dilation, dilation, dilation, dilation), mode='reflect').long().squeeze(0).squeeze(0)
    step = dilation
    neighbors = [
        padded[:-2*step, step:-step], padded[2*step:, step:-step],
        padded[step:-step, :-2*step], padded[step:-step, 2*step:],
        padded[:-2*step, :-2*step], padded[:-2*step, 2*step:],
        padded[2*step:, :-2*step], padded[2*step:, 2*step:]
    ]
    knn_idx = torch.stack(neighbors[:k], dim=-1).reshape(N, k)
    return knn_idx.unsqueeze(0)

def get_cached_knn(grid_size, device, k=8, dilation=1):
    """Returns the cached stencil for a given grid size and device."""
    key = (grid_size, str(device), k, dilation)
    if key not in _KNN_CACHE:
        _KNN_CACHE[key] = get_deterministic_grid_knn(grid_size, device, k, dilation)
    return _KNN_CACHE[key]

# ------------------------------------------------------------------------------
# Local SO(2) Frame with fallback
# ------------------------------------------------------------------------------
def compute_local_frames(grad_k_vec, coords, neighbor_idx, k_field, eps=1e-4):
    """
    Computes an orthonormal local frame (e1, e2) at each node.
    e1 is primarily aligned with the gradient of k. In regions where
    the gradient is small, a soft‑weighted neighbor direction is used.
    """
    B, N, _ = coords.shape
    K = neighbor_idx.shape[-1]
    device = coords.device

    grad_norm = torch.norm(grad_k_vec, dim=-1, keepdim=True)
    e1_grad = grad_k_vec / (grad_norm + 1e-8)

    # Field‑based fallback
    idx_flat = neighbor_idx.reshape(B, -1)
    k_j = torch.gather(k_field, 1, idx_flat).view(B, N, K)
    k_i = k_field.unsqueeze(2)
    delta_k = torch.abs(k_j - k_i)
    weights = F.softmax(delta_k * 10.0, dim=2)  # higher weight to similar k

    coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B, N, K, 2)
    delta_x = coords_j - coords.unsqueeze(2)
    dist = torch.norm(delta_x, dim=-1, keepdim=True) + 1e-8
    unit_dir = delta_x / dist

    ref_vec = torch.sum(weights.unsqueeze(-1) * unit_dir, dim=2)
    ref_norm = torch.norm(ref_vec, dim=-1, keepdim=True) + 1e-8
    e1_fallback = ref_vec / ref_norm

    # Blending factor: use gradient where strong, fallback where weak
    alpha = torch.sigmoid((grad_norm - eps) / (eps * 0.5))
    e1 = alpha * e1_grad + (1.0 - alpha) * e1_fallback
    e1 = e1 / (torch.norm(e1, dim=-1, keepdim=True) + 1e-8)

    e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)
    return e1, e2

# ------------------------------------------------------------------------------
# ISN Block
# ------------------------------------------------------------------------------
class ISNBlock(nn.Module):
    def __init__(self, hidden_dim=384):
        super().__init__()
        # h_i, h_j, mag, cos_phase, sin_phase, z_depth
        edge_in_dim = 2 * hidden_dim + 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h, coords, e1, e2, neighbor_idx, sigma, z_depth):
        B, N, C = h.shape
        K = neighbor_idx.shape[-1]

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B,N,K,2)
        h_j = torch.gather(h, 1, idx_flat.unsqueeze(-1).expand(-1,-1,C)).view(B,N,K,C)

        h_i = h.unsqueeze(2).expand(-1,-1,K,-1)
        coords_i = coords.unsqueeze(2).expand(-1,-1,K,-1)
        sigma_i = sigma.unsqueeze(2).expand(-1,-1,K,-1)

        delta_x = coords_j - coords_i
        dx_local = torch.sum(delta_x * e1.unsqueeze(2), dim=-1, keepdim=True)
        dy_local = torch.sum(delta_x * e2.unsqueeze(2), dim=-1, keepdim=True)

        mag = torch.sqrt(dx_local**2 + dy_local**2) / (sigma_i + 1e-8)
        phase = torch.atan2(dy_local, dx_local)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        z_depth_i = z_depth.unsqueeze(2).expand(-1,-1,K,-1)

        edge_input = torch.cat([h_i, h_j, mag, cos_phase, sin_phase, z_depth_i], dim=-1)
        messages = self.edge_mlp(edge_input)
        agg = messages.mean(dim=2)

        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(node_input)
        return h + self.res_scale * self.norm(h_new)

# ------------------------------------------------------------------------------
# AlienX Operator
# ------------------------------------------------------------------------------
class AlienXOperator(nn.Module):
    def __init__(self, hidden_dim=384, L=4, alpha=1.0, sigma_0=0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([ISNBlock(hidden_dim) for _ in range(L)])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, coords, k, grad_k_mag, grad_k_vec):
        """
        Args:
            coords: [B, N, 2] grid coordinates
            k:      [B, N] permeability field
            grad_k_mag: [B, N] magnitude of gradient of k
            grad_k_vec: [B, N, 2] gradient vector of k
        Returns:
            p_pred: [B, N] predicted pressure field
        """
        B, N, _ = coords.shape
        grid_size = int(math.sqrt(N))
        device = coords.device

        # Deterministic cached KNN
        neighbor_idx = get_cached_knn(grid_size, device, k=8, dilation=1)
        neighbor_idx = neighbor_idx.repeat(B, 1, 1)

        # Local scale sigma
        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B, N, -1, 2)
        coords_i = coords.unsqueeze(2)
        distances = torch.norm(coords_i - coords_j, dim=-1)
        sigma = distances.mean(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, N, 1]

        # Scale‑space depth
        z_depth = self.alpha * torch.log(sigma / self.sigma_0 + 1e-8)  # [B, N, 1]

        # Local SO(2) frame
        e1, e2 = compute_local_frames(grad_k_vec, coords, neighbor_idx, k)

        # Input projection
        grad_k_mag_norm = grad_k_mag * sigma.squeeze(-1)  # scale‑aware magnitude
        x_in = torch.cat([k.unsqueeze(-1), grad_k_mag_norm.unsqueeze(-1), z_depth], dim=-1)
        h = self.input_proj(x_in)

        # Message passing blocks
        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth)

        # Output
        p_pred = self.output_proj(h).squeeze(-1)  # [B, N]
        return p_pred