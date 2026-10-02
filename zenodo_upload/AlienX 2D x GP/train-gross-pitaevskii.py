

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler


# ═══════════════════════════════════════════════════════════════════════
# 1. ISOTROPIC 24-NEIGHBOR STENCIL
# ═══════════════════════════════════════════════════════════════════════
RADIUS = 2.8284271247461903
ISO_OFFSETS = []
for dr in range(-int(RADIUS), int(RADIUS) + 1):
    for dc in range(-int(RADIUS), int(RADIUS) + 1):
        if dr == 0 and dc == 0:
            continue
        if math.sqrt(dr * dr + dc * dc) <= RADIUS + 1e-8:
            ISO_OFFSETS.append((dr, dc))
K_ISO = len(ISO_OFFSETS)


def get_isotropic_knn(grid_size, device, dilation=1):
    N = grid_size * grid_size
    idx = torch.arange(N, device=device).view(1, 1, grid_size, grid_size)
    pad = 2 * dilation
    padded = F.pad(idx.float(), (pad, pad, pad, pad),
                   mode="reflect").long().squeeze(0).squeeze(0)
    nlist = []
    for dr, dc in ISO_OFFSETS:
        r0 = pad + dr * dilation
        c0 = pad + dc * dilation
        block = padded[r0:r0 + grid_size, c0:c0 + grid_size]
        nlist.append(block.reshape(-1))
    return torch.stack(nlist, dim=-1).reshape(N, K_ISO).unsqueeze(0)


_KNN_CACHE = {}


def get_cached_iso_knn(grid_size, device):
    key = (grid_size, str(device))
    if key not in _KNN_CACHE:
        dil = max(1, grid_size // 16)
        _KNN_CACHE[key] = get_isotropic_knn(grid_size, device, dilation=dil)
    return _KNN_CACHE[key]


_COORDS_CACHE = {}


def model_coords(N, device):
    key = (N, str(device))
    if key not in _COORDS_CACHE:
        x = torch.linspace(-1, 1, N, device=device)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        _COORDS_CACHE[key] = torch.stack(
            [X.flatten(), Y.flatten()], dim=-1).unsqueeze(0)
    return _COORDS_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════
# 2. GP SOLVER (split-step Fourier, 2D focusing)
# ═══════════════════════════════════════════════════════════════════════
def gp_initial_condition(N, device, seed=None, amp=1.0, sigma_frac=0.20,
                          phase_amp=0.4, n_modes=3):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.linspace(0, 2 * math.pi, N + 1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r2 = (X - math.pi) ** 2 + (Y - math.pi) ** 2
    sigma = sigma_frac * 2 * math.pi
    amp_field = amp * torch.exp(-r2 / (2 * sigma * sigma))
    phase = torch.zeros_like(amp_field)
    for k in range(1, n_modes + 1):
        phase += (phase_amp / k) * torch.sin(k * X + 0.7 * k) * \
                 torch.cos(k * Y - 1.1 * k)
    return amp_field * torch.exp(1j * phase)


def gp_step(psi, dt, g, k2):
    lin_half = torch.exp(-0.5j * dt * 0.5 * k2)
    psi_hat = torch.fft.fft2(psi) * lin_half
    psi = torch.fft.ifft2(psi_hat)
    dens = psi.real ** 2 + psi.imag ** 2
    psi = psi * torch.exp(-1j * g * dens * dt)
    psi_hat = torch.fft.fft2(psi) * lin_half
    return torch.fft.ifft2(psi_hat)


def gp_rollout(psi0, g, n_steps, dt, k2):
    psi = psi0
    frames = [psi.clone()]
    for _ in range(n_steps):
        psi = gp_step(psi, dt, g, k2)
        frames.append(psi.clone())
    return torch.stack(frames)


def generate_gp_pair(resolution, device, angle_deg=0.0, seed=None,
                     g=-1.0, dt=2e-3, n_steps=1):
    if seed is not None:
        torch.manual_seed(seed)
    N = resolution
    kx = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=device)
    ky = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2

    theta = angle_deg * math.pi / 180.0
    x = torch.linspace(0, 2 * math.pi, N + 1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="ij")
    c, s = math.cos(-theta), math.sin(-theta)
    Xr = c * (X - math.pi) - s * (Y - math.pi) + math.pi
    Yr = s * (X - math.pi) + c * (Y - math.pi) + math.pi
    r2 = (Xr - math.pi) ** 2 + (Yr - math.pi) ** 2
    sigma = 0.20 * 2 * math.pi
    amp = 1.0 * torch.exp(-r2 / (2 * sigma * sigma))
    phase = torch.zeros_like(amp)
    for k in range(1, 4):
        phase += (0.4 / k) * torch.sin(k * Xr + 0.7 * k) * \
                 torch.cos(k * Yr - 1.1 * k)
    psi0 = amp * torch.exp(1j * phase)

    frames = gp_rollout(psi0, g, n_steps, dt, k2)
    return frames[0], frames[n_steps]


# ═══════════════════════════════════════════════════════════════════════
# 3. ISN BLOCK (even-harmonic edge features)
# ═══════════════════════════════════════════════════════════════════════
class ISNBlock(nn.Module):
    def __init__(self, hidden_dim=128, k=K_ISO, sigma_kernel=1.2):
        super().__init__()
        self.k = k
        self.sigma_kernel = sigma_kernel
        edge_in_dim = 2 * hidden_dim + 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.angular_weight = nn.Linear(4, 1, bias=False)
        nn.init.normal_(self.angular_weight.weight, std=0.1)

    def forward(self, h, coords, e1, e2, neighbor_idx, sigma, z_depth):
        B, N, C = h.shape
        K = neighbor_idx.shape[-1]

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(
            coords, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 2)
        ).view(B, N, K, 2)
        h_j = torch.gather(
            h, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).view(B, N, K, C)

        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
        coords_i = coords.unsqueeze(2).expand(-1, -1, K, -1)
        sigma_i = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
        z_depth_i = z_depth.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)

        delta = coords_j - coords_i
        dx = torch.sum(delta * e1.unsqueeze(2), dim=-1, keepdim=True)
        dy = torch.sum(delta * e2.unsqueeze(2), dim=-1, keepdim=True)

        mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8) / (sigma_i + 1e-8)
        w_radial = torch.exp(-0.5 * (mag / self.sigma_kernel) ** 2)

        phase = torch.atan2(dy, dx)
        cos2 = torch.cos(2 * phase)
        sin2 = torch.sin(2 * phase)
        cos4 = torch.cos(4 * phase)
        sin4 = torch.sin(4 * phase)

        angular_feats = torch.cat([cos2, sin2, cos4, sin4], dim=-1)
        angular_gate = torch.sigmoid(self.angular_weight(angular_feats))
        total_weight = angular_gate * w_radial

        # THE FIX: even harmonics in edge features
        edge_input = torch.cat(
            [h_i, h_j, mag, cos2, sin2, z_depth_i], dim=-1
        )
        messages = self.edge_mlp(edge_input) * total_weight
        agg = messages.sum(dim=2) / (total_weight.sum(dim=2) + 1e-8)

        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h + self.res_scale * self.norm(h_new)


# ═══════════════════════════════════════════════════════════════════════
# 4. ALIENX GP OPERATOR
# ═══════════════════════════════════════════════════════════════════════
class AlienXGP(nn.Module):
    def __init__(self, hidden_dim=128, num_blocks=4, k=K_ISO,
                 sigma_kernel=1.2, alpha=0.5, sigma_0=0.1,
                 anchor_temperature=0.2, R_phys=0.25):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.anchor_temperature = anchor_temperature
        self.R_phys = R_phys

        self.input_mlp = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            ISNBlock(hidden_dim, k=k, sigma_kernel=sigma_kernel)
            for _ in range(num_blocks)
        ])
        self.output_linear = nn.Linear(hidden_dim, 2)

    def _spectral_grad(self, f, N):
        kx = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=f.device)
        ky = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=f.device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        f_hat = torch.fft.fft2(f)
        dfdx = torch.fft.ifft2(1j * KX * f_hat).real
        dfdy = torch.fft.ifft2(1j * KY * f_hat).real
        return torch.stack([dfdx, dfdy], dim=-1)

    def forward(self, psi_r, psi_i, coords, neighbor_idx, N):
        B = psi_r.shape[0]
        device = psi_r.device

        psi_r_g = psi_r.view(B, N, N)
        psi_i_g = psi_i.view(B, N, N)
        k_g = psi_r_g ** 2 + psi_i_g ** 2

        grad_k = self._spectral_grad(k_g, N)
        grad_psi_r = self._spectral_grad(psi_r_g, N)
        grad_psi_i = self._spectral_grad(psi_i_g, N)

        k = k_g.reshape(B, N * N)
        grad_k = grad_k.reshape(B, N * N, 2)
        grad_psi_r = grad_psi_r.reshape(B, N * N, 2)
        grad_psi_i = grad_psi_i.reshape(B, N * N, 2)
        grad_k_mag = grad_k.norm(dim=-1)

        grad_norm = grad_k / (grad_k_mag.unsqueeze(-1) + 1e-8)

        anchor_w = torch.softmax(k / self.anchor_temperature, dim=1)
        xw = coords * anchor_w.unsqueeze(-1)
        S = torch.einsum("bnd,bnc->bdc", xw, xw)
        S = S + 1e-6 * torch.eye(2, device=device).unsqueeze(0)
        _, evec = torch.linalg.eigh(S)
        ref_vec = evec[..., -1]
        ref_vec = ref_vec / (ref_vec.norm(dim=-1, keepdim=True) + 1e-8)
        ref_vec = ref_vec.unsqueeze(1).expand(-1, N * N, -1)

        gate = torch.sigmoid((grad_k_mag.unsqueeze(-1) - 0.1) / 0.05)
        e1 = gate * grad_norm + (1.0 - gate) * ref_vec
        e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
        e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)

        grad_psi_r_lx = (grad_psi_r * e1).sum(-1, keepdim=True)
        grad_psi_r_ly = (grad_psi_r * e2).sum(-1, keepdim=True)
        grad_psi_i_lx = (grad_psi_i * e1).sum(-1, keepdim=True)
        grad_psi_i_ly = (grad_psi_i * e2).sum(-1, keepdim=True)

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(
            coords, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 2)
        ).view(B, N * N, self.k, 2)
        dists = (coords.unsqueeze(2) - coords_j).norm(dim=-1)
        sigma = dists.mean(dim=-1).clamp(min=1e-6)

        z_val = self.alpha * math.log(self.R_phys / self.sigma_0 + 1e-8)
        z_depth = torch.full_like(k, z_val)

        x = torch.cat([
            k.unsqueeze(-1),
            grad_k_mag.unsqueeze(-1),
            grad_psi_r_lx, grad_psi_r_ly,
            grad_psi_i_lx, grad_psi_i_ly,
            z_depth.unsqueeze(-1),
        ], dim=-1)

        h = self.input_mlp(x)
        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth)

        delta = self.output_linear(h)
        return delta[..., 0], delta[..., 1]


# ═══════════════════════════════════════════════════════════════════════
# 5. DATASET
# ═══════════════════════════════════════════════════════════════════════
class GPDatasetGPU(Dataset):
    def __init__(self, resolutions, samples_per_res=16, device="cuda",
                 angles=None):
        self.device = torch.device(device if torch.cuda.is_available()
                                    and device == "cuda" else "cpu")
        self.data = []
        for res in resolutions:
            for _ in range(samples_per_res):
                angle = (float(np.random.uniform(0, 360))
                         if angles is None else float(np.random.choice(angles)))
                seed = int(np.random.randint(0, 1_000_000))
                psi_t, psi_tp = generate_gp_pair(res, self.device,
                                                 angle_deg=angle, seed=seed,
                                                 n_steps=1)
                self.data.append((psi_t.real.flatten(),
                                  psi_t.imag.flatten(),
                                  psi_tp.real.flatten(),
                                  psi_tp.imag.flatten(),
                                  res))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, batch_size_map):
        self.dataset = dataset
        self.bs_map = batch_size_map
        self.groups = {}
        for i, item in enumerate(dataset.data):
            self.groups.setdefault(item[4], []).append(i)

    def __iter__(self):
        order = list(self.groups.keys())
        np.random.shuffle(order)
        for res in order:
            idx = self.groups[res].copy()
            np.random.shuffle(idx)
            bs = self.bs_map.get(res, 1)
            for i in range(0, len(idx), bs):
                yield idx[i:i + bs]

    def __len__(self):
        return sum((len(v) + self.bs_map.get(k, 1) - 1)
                   // self.bs_map.get(k, 1)
                   for k, v in self.groups.items())


def collate_gp(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            batch[0][4])


def interior_mse(pred, target, N):
    B = pred.shape[0]
    p = pred.view(B, N, N)
    t = target.view(B, N, N)
    d = max(1, N // 16)
    crop = 2 * d
    return F.mse_loss(p[:, crop:-crop, crop:-crop],
                      t[:, crop:-crop, crop:-crop])


# ═══════════════════════════════════════════════════════════════════════
# 6. TRAINING
# ═══════════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = AlienXGP(hidden_dim=args.hidden_dim,
                     num_blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    resolutions = [16, 32, 64]
    print("generating dataset...")
    dataset = GPDatasetGPU(resolutions, samples_per_res=24, device="cuda")
    bs_map = {16: 8, 32: 4, 64: 2}
    sampler = DynamicBatchSampler(dataset, bs_map)
    dl = DataLoader(dataset, batch_sampler=sampler,
                    collate_fn=collate_gp, num_workers=0)

    best = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        model.train()

        for psi_r, psi_i, t_r, t_i, res in dl:
            psi_r = psi_r.to(device); psi_i = psi_i.to(device)
            t_r = t_r.to(device); t_i = t_i.to(device)

            knn = get_cached_iso_knn(res, device).repeat(psi_r.shape[0], 1, 1)
            coords = model_coords(res, device).repeat(psi_r.shape[0], 1, 1)

            d_r, d_i = model(psi_r, psi_i, coords, knn, res)
            pred_r = psi_r + d_r
            pred_i = psi_i + d_i

            loss = interior_mse(pred_r, t_r, res) + \
                   interior_mse(pred_i, t_i, res)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if torch.isnan(loss):
                print(f"NaN at epoch {epoch}. Reloading best.")
                if os.path.exists("alienx_gp_best.pt"):
                    model.load_state_dict(torch.load(
                        "alienx_gp_best.pt", map_location=device))
                return

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), "alienx_gp_best.pt")

        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  loss={avg:.6f}  best={best:.6f}  "
                  f"lr={sched.get_last_lr()[0]:.1e}  "
                  f"({time.time()-t0:.0f}s)")

    torch.save(model.state_dict(), "alienx_gp_final.pt")
    print(f"\nbest loss: {best:.6f}")
    print(f"total time: {(time.time()-t0)/60:.1f} min")


# ═══════════════════════════════════════════════════════════════════════
# 7. EVALUATION
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_rotation(model, device, res=64):
    print(f"\nRotation equivariance ({res}×{res}):")
    errs = []
    model.eval()
    for angle in range(0, 360, 15):
        psi_t, psi_tp = generate_gp_pair(res, device, angle_deg=angle,
                                          seed=42, n_steps=1)
        psi_r = psi_t.real.flatten().unsqueeze(0)
        psi_i = psi_t.imag.flatten().unsqueeze(0)
        t_r = psi_tp.real.flatten().unsqueeze(0)
        t_i = psi_tp.imag.flatten().unsqueeze(0)
        knn = get_cached_iso_knn(res, device)
        coords = model_coords(res, device)

        d_r, d_i = model(psi_r, psi_i, coords, knn, res)
        pred_r = psi_r + d_r
        pred_i = psi_i + d_i

        err = 0.5 * (interior_mse(pred_r, t_r, res).item() +
                     interior_mse(pred_i, t_i, res).item())
        errs.append(err)
        print(f"  {angle:3d}°  MSE={err:.6f}")

    errs = np.array(errs)
    print(f"  mean={errs.mean():.6f}  std={errs.std():.6f}  "
          f"max={errs.max():.6f}")


@torch.no_grad()
def eval_scale(model, device):
    print("\nScale invariance (zero-shot):")
    model.eval()
    for res in [16, 32, 64, 128]:
        psi_t, psi_tp = generate_gp_pair(res, device, angle_deg=0.0,
                                          seed=123, n_steps=1)
        psi_r = psi_t.real.flatten().unsqueeze(0)
        psi_i = psi_t.imag.flatten().unsqueeze(0)
        t_r = psi_tp.real.flatten().unsqueeze(0)
        t_i = psi_tp.imag.flatten().unsqueeze(0)
        knn = get_cached_iso_knn(res, device)
        coords = model_coords(res, device)

        d_r, d_i = model(psi_r, psi_i, coords, knn, res)
        pred_r = psi_r + d_r
        pred_i = psi_i + d_i
        err = 0.5 * (interior_mse(pred_r, t_r, res).item() +
                     interior_mse(pred_i, t_i, res).item())
        print(f"  {res}×{res}: {err:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN — Colab-safe argparse
# ═══════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    # parse_known_args swallows Colab's kernel args (-f /path/to.json)
    args, _ = ap.parse_known_args()
    return args


def main():
    args = parse_args()
    train(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlienXGP(hidden_dim=args.hidden_dim,
                     num_blocks=args.blocks).to(device)
    model.load_state_dict(torch.load("alienx_gp_best.pt", map_location=device))
    model.eval()

    eval_scale(model, device)
    eval_rotation(model, device, res=64)


if __name__ == "__main__":
    main()