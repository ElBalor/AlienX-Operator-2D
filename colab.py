"""
AlienX (ISN) - 2D Release v2: Self-Contained Colab Script
=========================================================
Single-file version: paste into a Colab cell (or run `python colab.py`).
Identical to the modular model.py / data.py / train.py, and identical to the
script that produced the v2 numbers. Single T4 GPU, ~15 min for 500 epochs.


"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler

# -- GLOBAL CONSTANTS ---------------------------------------------------------
RADIUS = 2.8284271247461903   # sqrt(8) - isotropic circular boundary, no cardinal spikes

ISO_OFFSETS = []
for dr in range(-int(RADIUS), int(RADIUS)+1):
    for dc in range(-int(RADIUS), int(RADIUS)+1):
        if dr == 0 and dc == 0:
            continue
        if math.sqrt(dr*dr + dc*dc) <= RADIUS + 1e-8:
            ISO_OFFSETS.append((dr, dc))

K_ISO = len(ISO_OFFSETS)
print(f"Isotropic stencil neighbors: {K_ISO}")


def get_isotropic_knn(grid_size, device, dilation=1):
    N = grid_size * grid_size
    idx = torch.arange(N, device=device).view(1, 1, grid_size, grid_size)
    pad = 2 * dilation
    padded = F.pad(idx.float(), (pad, pad, pad, pad), mode='reflect').long().squeeze(0).squeeze(0)

    neighbor_list = []
    for dr, dc in ISO_OFFSETS:
        r_start = pad + dr * dilation
        c_start = pad + dc * dilation
        block = padded[r_start:r_start+grid_size, c_start:c_start+grid_size]
        neighbor_list.append(block.reshape(-1))

    knn_idx = torch.stack(neighbor_list, dim=-1).reshape(N, K_ISO)
    return knn_idx.unsqueeze(0)


KNN_CACHE = {}


def get_cached_iso_knn(grid_size, device):
    key = (grid_size, str(device))
    if key not in KNN_CACHE:
        scale_factor = max(1, grid_size // 16)
        KNN_CACHE[key] = get_isotropic_knn(grid_size, device, dilation=scale_factor)
    return KNN_CACHE[key]


# -- GPU DATA GENERATION ------------------------------------------------------
def generate_darcy_sample_gpu(resolution, device, angle_deg=0.0, seed=None):
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


# -- ISN BLOCK WITH ISOTROPIC RADIAL TAPER AND EVEN HARMONICS -----------------
class ISNBlock(nn.Module):
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
        angular_in = 4
        self.angular_weight = nn.Linear(angular_in, 1, bias=False)
        nn.init.normal_(self.angular_weight.weight, mean=0.0, std=0.1)

    def forward(self, h, coords, e1, e2, neighbor_idx, sigma, z_depth):
        B, N, C = h.shape
        K = neighbor_idx.shape[-1]

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B,N,K,2)
        h_j = torch.gather(h, 1, idx_flat.unsqueeze(-1).expand(-1,-1,C)).view(B,N,K,C)

        h_i = h.unsqueeze(2).expand(-1,-1,K,-1)
        coords_i = coords.unsqueeze(2).expand(-1,-1,K,-1)

        sigma_i = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
        z_depth_i = z_depth.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)

        delta_x = coords_j - coords_i
        dx_local = torch.sum(delta_x * e1.unsqueeze(2), dim=-1, keepdim=True)
        dy_local = torch.sum(delta_x * e2.unsqueeze(2), dim=-1, keepdim=True)

        mag = torch.sqrt(dx_local**2 + dy_local**2 + 1e-8) / (sigma_i + 1e-8)

        w_radial = torch.exp(-0.5 * (mag / self.sigma_kernel)**2)

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
        angular_feats = torch.cat(angular_feats, dim=-1)  # (B,N,K,4)

        angular_gate = torch.sigmoid(self.angular_weight(angular_feats))
        total_weight = angular_gate * w_radial

        edge_input = torch.cat([h_i, h_j, mag, cos_phase, sin_phase, z_depth_i], dim=-1)
        messages = self.edge_mlp(edge_input) * total_weight
        agg = messages.sum(dim=2) / (total_weight.sum(dim=2) + 1e-8)

        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(node_input)
        return h + self.res_scale * self.norm(h_new)


# -- ALIENX OPERATOR WITH INERTIA TENSOR ANCHOR & CONSTANT DEPTH --------------
class AlienXOperator(nn.Module):
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
        B, N, _ = coords.shape
        device = coords.device

        # Spatial inertia tensor anchor (second moment) + sign_lock (ported from
        # the 3D QSA_ISNBlock3D): eigh's eigenvector SIGN is unspecified under
        # rotation (measured dot(R e1, e1_rot) = -1.0000 at 90/180 deg), which
        # rotates the local frame phase by pi and breaks the odd cos/sin edge
        # features. Anchor the sign to the k-weighted centroid m (a covariant
        # data vector, the 2D analogue of the 3D Hn_tan anchor): s = <v, m> is
        # odd in v (cancels eigh's arbitrary flip) and D4-invariant (-1 in D4
        # flips BOTH signs), so e1 = s * v is a pure function of the data.
        # No new parameters - checkpoints stay compatible.
        anchor_weights = torch.softmax(k / self.anchor_temperature, dim=1)  # (B,N)
        m = torch.einsum('bn,bnd->bd', anchor_weights, coords)              # (B,2)
        x_weighted = coords * anchor_weights.unsqueeze(-1)  # (B,N,2)
        S = torch.einsum('bnd,bnc->bdc', x_weighted, x_weighted)  # (B,2,2)
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        ref_vec = eigenvectors[..., -1]  # leading eigenvector, (B,2)
        ref_vec = ref_vec / (ref_vec.norm(dim=-1, keepdim=True) + 1e-8)
        ref_vec = ref_vec.unsqueeze(1).expand(-1, N, -1)  # (B,N,2)

        # sign_lock: deterministic, equivariant eigenvector sign
        sign_lock = torch.sign((ref_vec * m.unsqueeze(1)).sum(-1, keepdim=True))  # (B,N,1)
        sign_lock = torch.where(sign_lock == 0, torch.ones_like(sign_lock), sign_lock)
        ref_vec = ref_vec * sign_lock

        grad_norm = grad_k / (grad_k_mag.unsqueeze(-1) + 1e-8)
        use_radial = (grad_k_mag.unsqueeze(-1) < self.fallback_threshold).float()

        e1 = grad_norm * (1 - use_radial) + ref_vec * use_radial
        e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
        e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)

        grid_size = int(math.sqrt(N))

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B,N,self.k,2)
        distances = torch.norm(coords.unsqueeze(2) - coords_j, dim=-1)

        sigma = distances.mean(dim=-1).clamp(min=1e-6)  # physical neighbor distance

        # Constant physical scale depth
        z_depth_value = self.alpha * torch.log(
            torch.tensor(self.R_phys / self.sigma_0 + 1e-8, dtype=torch.float32, device=device)
        )
        z_depth = z_depth_value.expand_as(k)  # (B,N)

        x = torch.stack([k, grad_k_mag, z_depth], dim=-1)
        h = self.input_mlp(x)

        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth)

        out = self.output_linear(h)
        return out.squeeze(-1)


# -- GPU DATASET (Pre-generated on GPU) ---------------------------------------
class DarcyDatasetGPU(Dataset):
    def __init__(self, resolutions, samples_per_res=16, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.data = []
        for res in resolutions:
            for _ in range(samples_per_res):
                angle = np.random.uniform(0.0, 360.0)
                seed = np.random.randint(0, 1000000)
                coords, k, grad_k_mag, grad_k_vec, p = generate_darcy_sample_gpu(res, self.device, angle, seed)
                self.data.append((coords, k, grad_k_mag, grad_k_vec, p, res))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, batch_size_map):
        self.dataset = dataset
        self.batch_size_map = batch_size_map
        self.groups = {}
        for i, (_, _, _, _, _, res) in enumerate(dataset.data):
            self.groups.setdefault(res, []).append(i)

    def __iter__(self):
        res_order = list(self.groups.keys())
        np.random.shuffle(res_order)
        for res in res_order:
            indices = self.groups[res].copy()
            np.random.shuffle(indices)
            batch_size = self.batch_size_map.get(res, 1)
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i+batch_size]
                if len(batch) > 0:
                    yield batch

    def __len__(self):
        total = 0
        for res, indices in self.groups.items():
            batch_size = self.batch_size_map.get(res, 1)
            total += (len(indices) + batch_size - 1) // batch_size
        return total


def collate_fn_gpu(batch):
    coords = torch.stack([b[0] for b in batch])
    k = torch.stack([b[1] for b in batch])
    grad_mag = torch.stack([b[2] for b in batch])
    grad_vec = torch.stack([b[3] for b in batch])
    p = torch.stack([b[4] for b in batch])
    res = batch[0][5]
    return coords, k, grad_mag, grad_vec, p, res


# Boundary crop scaled by dilation
def compute_interior_loss(pred, target, grid_size):
    B = pred.shape[0]
    p_grid = pred.view(B, grid_size, grid_size)
    t_grid = target.view(B, grid_size, grid_size)
    dilation = max(1, grid_size // 16)
    crop = 2 * dilation
    p_interior = p_grid[:, crop:-crop, crop:-crop]
    t_interior = t_grid[:, crop:-crop, crop:-crop]
    return F.mse_loss(p_interior, t_interior)


# -- TRAINING -----------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlienXOperator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

    resolutions = [16, 32, 64, 128]
    dataset = DarcyDatasetGPU(resolutions, samples_per_res=16, device='cuda')

    batch_size_map = {16: 8, 32: 4, 64: 2, 128: 1}
    batch_sampler = DynamicBatchSampler(dataset, batch_size_map)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn_gpu, num_workers=0)

    best_loss = float('inf')

    for epoch in range(500):
        total_loss = 0.0
        nan_detected = False
        model.train()

        for batch in dataloader:
            coords, k, grad_mag, grad_vec, p, res = batch
            neighbor_idx = get_cached_iso_knn(res, device).repeat(coords.shape[0], 1, 1)

            pred = model(coords, k, grad_vec, grad_mag, neighbor_idx)
            loss = compute_interior_loss(pred, p, res)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if torch.isnan(loss).item():
                print(f"NaN detected at epoch {epoch}. Reverting to best model.")
                nan_detected = True
                break

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "alienx_best.pt")

        if nan_detected:
            if os.path.exists("alienx_best.pt"):
                model.load_state_dict(torch.load("alienx_best.pt", map_location=device))
            break

        scheduler.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "alienx_model.pt")
    print("Model saved successfully.")


# -- EVALUATION ---------------------------------------------------------------
def evaluate_scale(model, device):
    print("\nZero-Shot Scale Invariance (Interior L1):")
    for res in [16, 32, 64, 128, 256]:
        coords, k, grad_mag, grad_vec, p = generate_darcy_sample_gpu(res, device, angle_deg=0, seed=123)
        coords = coords.unsqueeze(0)
        k = k.unsqueeze(0)
        grad_mag = grad_mag.unsqueeze(0)
        grad_vec = grad_vec.unsqueeze(0)
        p = p.unsqueeze(0)

        neighbor_idx = get_cached_iso_knn(res, device)

        with torch.no_grad():
            pred = model(coords, k, grad_vec, grad_mag, neighbor_idx)

        p_grid = pred.view(1, res, res)
        t_grid = p.view(1, res, res)
        # match training loss cropping
        dilation = max(1, res // 16)
        crop = 2 * dilation
        err = F.l1_loss(p_grid[:, crop:-crop, crop:-crop], t_grid[:, crop:-crop, crop:-crop]).item()
        print(f"{res}x{res}: {err:.6f}")


def evaluate_rotation(model, device):
    print("\nRotation Equivariance (Interior L1):")
    for res in [16, 32, 64, 128, 256]:
        print(f"\nResolution {res}x{res}:")
        for angle in [0, 45, 90, 180, 270]:
            coords, k, grad_mag, grad_vec, p = generate_darcy_sample_gpu(res, device, angle_deg=angle, seed=123)
            coords = coords.unsqueeze(0)
            k = k.unsqueeze(0)
            grad_mag = grad_mag.unsqueeze(0)
            grad_vec = grad_vec.unsqueeze(0)
            p = p.unsqueeze(0)

            neighbor_idx = get_cached_iso_knn(res, device)

            with torch.no_grad():
                pred = model(coords, k, grad_vec, grad_mag, neighbor_idx)

            p_grid = pred.view(1, res, res)
            t_grid = p.view(1, res, res)
            dilation = max(1, res // 16)
            crop = 2 * dilation
            err = F.l1_loss(p_grid[:, crop:-crop, crop:-crop], t_grid[:, crop:-crop, crop:-crop]).item()
            print(f"  {angle}: {err:.6f}")


def evaluate_arbitrary(model, device):
    angles = [13.0, 27.0, 77.0, 123.0, 199.0]
    print("\nArbitrary Continuous Angle Equivariance Test (Interior L1):")
    for res in [16, 32, 64, 128, 256]:
        print(f"\n--- Resolution {res}x{res} ---")
        # baseline
        coords, k, grad_mag, grad_vec, p = generate_darcy_sample_gpu(res, device, angle_deg=0, seed=123)
        coords = coords.unsqueeze(0)
        k = k.unsqueeze(0)
        grad_mag = grad_mag.unsqueeze(0)
        grad_vec = grad_vec.unsqueeze(0)
        p = p.unsqueeze(0)
        neighbor_idx = get_cached_iso_knn(res, device)

        with torch.no_grad():
            pred = model(coords, k, grad_vec, grad_mag, neighbor_idx)
        dilation = max(1, res // 16)
        crop = 2 * dilation
        baseline_err = F.l1_loss(pred.view(1, res, res)[:, crop:-crop, crop:-crop],
                                 p.view(1, res, res)[:, crop:-crop, crop:-crop]).item()
        print(f"  0.0 (Baseline) : {baseline_err:.6f}")

        for angle in angles:
            coords, k, grad_mag, grad_vec, p = generate_darcy_sample_gpu(res, device, angle_deg=angle, seed=123)
            coords = coords.unsqueeze(0)
            k = k.unsqueeze(0)
            grad_mag = grad_mag.unsqueeze(0)
            grad_vec = grad_vec.unsqueeze(0)
            p = p.unsqueeze(0)

            with torch.no_grad():
                pred = model(coords, k, grad_vec, grad_mag, neighbor_idx)

            err = F.l1_loss(pred.view(1, res, res)[:, crop:-crop, crop:-crop],
                            p.view(1, res, res)[:, crop:-crop, crop:-crop]).item()
            delta = abs(err - baseline_err)
            print(f"  {angle:5.1f}          : {err:.6f}  (Delta vs 0 = {delta:.6f})")


def plot_continuous_rotation(model, device, res=64):
    angles = list(range(0, 360, 1))
    errors = []
    neighbor_idx = get_cached_iso_knn(res, device)

    dilation = max(1, res // 16)
    crop = 2 * dilation

    for angle in angles:
        coords, k, grad_mag, grad_vec, p = generate_darcy_sample_gpu(res, device, angle_deg=angle, seed=123)
        coords = coords.unsqueeze(0)
        k = k.unsqueeze(0)
        grad_mag = grad_mag.unsqueeze(0)
        grad_vec = grad_vec.unsqueeze(0)
        p = p.unsqueeze(0)

        with torch.no_grad():
            pred = model(coords, k, grad_vec, grad_mag, neighbor_idx)

        err = F.l1_loss(pred.view(1, res, res)[:, crop:-crop, crop:-crop],
                        p.view(1, res, res)[:, crop:-crop, crop:-crop]).item()
        errors.append(err)

    plt.figure(figsize=(10,5), facecolor='#0a0f1c')
    plt.plot(angles, errors, color='#f4a261', linewidth=1.5)
    plt.title(f"AlienX Continuous Rotation Error ({res}x{res})", color='white', fontsize=14)
    plt.xlabel("Rotation Angle (degrees)", color='white')
    plt.ylabel("Interior L1 Error", color='white')
    plt.grid(True, color='#1a2b4c', alpha=0.5)
    plt.gca().set_facecolor('#0a0f1c')
    plt.gca().tick_params(colors='white')
    plt.tight_layout()
    plt.savefig("alienx_rotation_continuous.png", dpi=200, facecolor='#0a0f1c')
    plt.show()
    print("Plot saved to alienx_rotation_continuous.png")


if __name__ == "__main__":
    train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlienXOperator().to(device)
    model.load_state_dict(torch.load("alienx_best.pt", map_location=device))
    model.eval()
    evaluate_scale(model, device)
    evaluate_rotation(model, device)
    evaluate_arbitrary(model, device)
    plot_continuous_rotation(model, device, res=64)
