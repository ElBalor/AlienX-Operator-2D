"""
AlienX (ISN) - 2D Release v2: Isotropic 24-Neighbor Stencil
============================================================
Isomorphic Spatial Net operating on continuous geometric manifolds
instead of fixed grids. Native rotation and scale invariance through:

  - local SO(2) frame construction (gradient primary, inertia-tensor fallback)
  - scale-normalized displacements (physical support radius constant)
  - 2*pi-periodic complex harmonic embedding with even harmonics
    (cos 2t, sin 2t, cos 4t, sin 4t) - message weights invariant under
    theta -> theta + pi
  - isotropic 24-neighbor stencil (radius sqrt(8), no cardinal spikes)
  - constant physical scale depth - scale-blindness by construction

    The grid is dead. The manifold is awake.

Eric Yaka (Elbalor / The Digital Necromancer)
Capital Software - Grimoire of Elbalor
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# ISOTROPIC 24-NEIGHBOR STENCIL
# Circular neighborhood of radius sqrt(8): uniform angular coverage, no
# cardinal-direction spikes, no 45-degree residual. Dilation scales with
# resolution so the physical support radius stays constant.
# ==============================================================================

RADIUS = 2.8284271247461903   # sqrt(8) - isotropic circular boundary, no cardinal spikes

ISO_OFFSETS = []
for dr in range(-int(RADIUS), int(RADIUS) + 1):
    for dc in range(-int(RADIUS), int(RADIUS) + 1):
        if dr == 0 and dc == 0:
            continue
        if math.sqrt(dr * dr + dc * dc) <= RADIUS + 1e-8:
            ISO_OFFSETS.append((dr, dc))

K_ISO = len(ISO_OFFSETS)   # 24


def get_isotropic_knn(grid_size, device, dilation=1):
    """Deterministic isotropic stencil over a grid, reflection-padded."""
    N = grid_size * grid_size
    idx = torch.arange(N, device=device).view(1, 1, grid_size, grid_size)
    pad = 2 * dilation
    padded = F.pad(idx.float(), (pad, pad, pad, pad), mode='reflect').long().squeeze(0).squeeze(0)

    neighbor_list = []
    for dr, dc in ISO_OFFSETS:
        r_start = pad + dr * dilation
        c_start = pad + dc * dilation
        block = padded[r_start:r_start + grid_size, c_start:c_start + grid_size]
        neighbor_list.append(block.reshape(-1))

    knn_idx = torch.stack(neighbor_list, dim=-1).reshape(N, K_ISO)
    return knn_idx.unsqueeze(0)


KNN_CACHE = {}


def get_cached_iso_knn(grid_size, device):
    """Cached stencil with dynamic dilation: physical support radius constant across resolutions."""
    key = (grid_size, str(device))
    if key not in KNN_CACHE:
        scale_factor = max(1, grid_size // 16)
        KNN_CACHE[key] = get_isotropic_knn(grid_size, device, dilation=scale_factor)
    return KNN_CACHE[key]


# ==============================================================================
# ISN BLOCK - isotropic radial taper + even harmonics
# ==============================================================================

class ISNBlock(nn.Module):
    """
    One message-passing block.

    Edge features: [h_i, h_j, mag, cos(phase), sin(phase), z_depth]
    Message weight: angular_gate (even harmonics) * radial taper (Gaussian)
    Aggregation: weighted mean over the 24 isotropic neighbors.
    Node update: MLP + residual (res_scale zero-init) + LayerNorm.
    """

    def __init__(self, hidden_dim=128, k=K_ISO, sigma_kernel=1.2):
        super().__init__()
        self.k = k
        self.sigma_kernel = sigma_kernel

        edge_in_dim = 2 * hidden_dim + 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.zeros(1))

        # Even harmonics only: cos(2t), sin(2t), cos(4t), sin(4t)
        # Invariant under t -> t + pi - the source of the 180-symmetric gating.
        angular_in = 4
        self.angular_weight = nn.Linear(angular_in, 1, bias=False)
        nn.init.normal_(self.angular_weight.weight, mean=0.0, std=0.1)

    def forward(self, h, coords, e1, e2, neighbor_idx, sigma, z_depth):
        B, N, C = h.shape
        K = neighbor_idx.shape[-1]

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 2)).view(B, N, K, 2)
        h_j = torch.gather(h, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C)).view(B, N, K, C)

        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
        coords_i = coords.unsqueeze(2).expand(-1, -1, K, -1)

        sigma_i = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
        z_depth_i = z_depth.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)

        delta_x = coords_j - coords_i
        dx_local = torch.sum(delta_x * e1.unsqueeze(2), dim=-1, keepdim=True)
        dy_local = torch.sum(delta_x * e2.unsqueeze(2), dim=-1, keepdim=True)

        mag = torch.sqrt(dx_local ** 2 + dy_local ** 2 + 1e-8) / (sigma_i + 1e-8)

        w_radial = torch.exp(-0.5 * (mag / self.sigma_kernel) ** 2)

        phase = torch.atan2(dy_local, dx_local)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Even harmonics: invariant under t -> t + pi
        angular_feats = [
            torch.cos(2 * phase),
            torch.sin(2 * phase),
            torch.cos(4 * phase),
            torch.sin(4 * phase)
        ]
        angular_feats = torch.cat(angular_feats, dim=-1)  # (B, N, K, 4)

        angular_gate = torch.sigmoid(self.angular_weight(angular_feats))
        total_weight = angular_gate * w_radial

        edge_input = torch.cat([h_i, h_j, mag, cos_phase, sin_phase, z_depth_i], dim=-1)
        messages = self.edge_mlp(edge_input) * total_weight
        agg = messages.sum(dim=2) / (total_weight.sum(dim=2) + 1e-8)

        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(node_input)
        return h + self.res_scale * self.norm(h_new)


# ==============================================================================
# ALIENX OPERATOR - inertia tensor anchor + constant physical depth
# ==============================================================================

class AlienXOperator(nn.Module):
    """
    Full ISN operator.

    - Local frame (e1, e2): e1 from grad(k) where the gradient is strong,
      inertia-tensor fallback (leading eigenvector of the k-weighted second
      moment) where it is weak. Both branches are individually SO(2)-equivariant.
    - sigma: mean physical neighbor distance - the local scale.
    - z_depth: CONSTANT physical scale depth alpha * ln(R_phys / sigma_0).
      Scale-blindness by construction - the source of zero-shot resolution transfer.
    - Input features: [k, grad_k_mag, z_depth] -> hidden -> L ISN blocks -> scalar.
    """

    def __init__(self, hidden_dim=128, num_blocks=4, k=K_ISO, sigma_kernel=1.2,
                 alpha=0.5, sigma_0=0.1, anchor_temperature=0.2, R_phys=0.25):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.hidden_dim = hidden_dim
        self.anchor_temperature = anchor_temperature
        self.R_phys = R_phys  # constant physical stencil radius (2 pixels at base resolution)
        self.fallback_threshold = 0.1  # |grad k| below which the inertia anchor takes the frame

        self.input_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            ISNBlock(hidden_dim, k=k, sigma_kernel=sigma_kernel)
            for _ in range(num_blocks)
        ])
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, coords, k, grad_k, grad_k_mag, neighbor_idx):
        """
        Args:
            coords:     [B, N, 2]   grid coordinates in [-1, 1]^2
            k:          [B, N]      permeability field
            grad_k:     [B, N, 2]   gradient vector of k (pushed forward)
            grad_k_mag: [B, N]      magnitude of grad k
            neighbor_idx: [1, N, K] cached isotropic stencil (from get_cached_iso_knn)
        Returns:
            p_pred: [B, N] predicted pressure field
        """
        B, N, _ = coords.shape
        device = coords.device

        # Spatial inertia tensor anchor (second moment).
        #
        # sign_lock (ported from the 3D QSA_ISNBlock3D): torch.linalg.eigh's
        # eigenvector SIGN is unspecified under rotation (measured
        # dot(R e1, e1_rot) = -1.0000 at 90/180 deg), which rotates the local
        # frame phase by pi and breaks the odd cos/sin edge features. The 2D
        # lock anchors the eigenvector's sign to the k-weighted centroid
        # direction m - a covariant data vector, the 2D analogue of the 3D
        # Hn_tan anchor. s = <v, m> is odd in v (cancels eigh's arbitrary
        # flip) and D4-invariant (m is covariant and -1 in D4 flips BOTH
        # signs), so e1 = s * v is a pure function of the data:
        # sign-deterministic and equivariant. No new parameters -
        # checkpoints stay compatible.
        anchor_weights = torch.softmax(k / self.anchor_temperature, dim=1)  # (B, N)
        m = torch.einsum('bn,bnd->bd', anchor_weights, coords)              # (B, 2) k-weighted centroid
        x_weighted = coords * anchor_weights.unsqueeze(-1)  # (B, N, 2)
        S = torch.einsum('bnd,bnc->bdc', x_weighted, x_weighted)  # (B, 2, 2)
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        ref_vec = eigenvectors[..., -1]  # leading eigenvector, (B, 2)
        ref_vec = ref_vec / (ref_vec.norm(dim=-1, keepdim=True) + 1e-8)
        ref_vec = ref_vec.unsqueeze(1).expand(-1, N, -1)  # (B, N, 2)

        # sign_lock: deterministic, equivariant eigenvector sign
        sign_lock = torch.sign((ref_vec * m.unsqueeze(1)).sum(-1, keepdim=True))  # (B, N, 1)
        sign_lock = torch.where(sign_lock == 0, torch.ones_like(sign_lock), sign_lock)
        ref_vec = ref_vec * sign_lock

        grad_norm = grad_k / (grad_k_mag.unsqueeze(-1) + 1e-8)
        use_radial = (grad_k_mag.unsqueeze(-1) < self.fallback_threshold).float()

        e1 = grad_norm * (1 - use_radial) + ref_vec * use_radial
        e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
        e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)

        grid_size = int(math.sqrt(N))

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 2)).view(B, N, self.k, 2)
        distances = torch.norm(coords.unsqueeze(2) - coords_j, dim=-1)

        sigma = distances.mean(dim=-1).clamp(min=1e-6)  # physical neighbor distance

        # Constant physical scale depth
        z_depth_value = self.alpha * torch.log(
            torch.tensor(self.R_phys / self.sigma_0 + 1e-8, dtype=torch.float32, device=device)
        )
        z_depth = z_depth_value.expand_as(k)  # (B, N)

        x = torch.stack([k, grad_k_mag, z_depth], dim=-1)
        h = self.input_mlp(x)

        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth)

        out = self.output_linear(h)
        return out.squeeze(-1)
