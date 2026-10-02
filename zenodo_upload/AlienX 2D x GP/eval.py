"""
AlienX 2D × GP — Full evaluation v3
=====================================

Loads alienx_gp_v3_best.pt (real IC diversity + K=5 rollout trained).
Runs 5 tests, cache-safe, no OOM.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def get_cached_iso_knn(grid_size, device):
    return get_isotropic_knn(grid_size, device,
                             dilation=max(1, grid_size // 16))


def model_coords(N, device):
    x = torch.linspace(-1, 1, N, device=device)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.flatten(), Y.flatten()], dim=-1).unsqueeze(0)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gp_step(psi, dt, g, k2):
    lin_half = torch.exp(-0.5j * dt * 0.5 * k2)
    psi_hat = torch.fft.fft2(psi) * lin_half
    psi = torch.fft.ifft2(psi_hat)
    dens = psi.real ** 2 + psi.imag ** 2
    psi = psi * torch.exp(-1j * g * dens * dt)
    psi_hat = torch.fft.fft2(psi) * lin_half
    return torch.fft.ifft2(psi_hat)


def generate_gp_pair(resolution, device, angle_deg=0.0, seed=None,
                     g=-1.0, dt=2e-3, n_steps=1):
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

    if seed is not None:
        g_ic = torch.Generator(device=device).manual_seed(seed)
        dx = 0.25 * (torch.rand(1, generator=g_ic, device=device).item() - 0.5)
        dy = 0.25 * (torch.rand(1, generator=g_ic, device=device).item() - 0.5)
        c1 = 0.4 * (0.7 + 0.6 * torch.rand(1, generator=g_ic, device=device).item())
        c2 = 0.4 * (0.7 + 0.6 * torch.rand(1, generator=g_ic, device=device).item())
        c3 = 0.4 * (0.7 + 0.6 * torch.rand(1, generator=g_ic, device=device).item())
        p1 = 2.0 * math.pi * torch.rand(1, generator=g_ic, device=device).item()
        p2 = 2.0 * math.pi * torch.rand(1, generator=g_ic, device=device).item()
        p3 = 2.0 * math.pi * torch.rand(1, generator=g_ic, device=device).item()
    else:
        dx = dy = 0.0
        c1, c2, c3 = 0.4, 0.2, 0.133
        p1, p2, p3 = 0.7, 1.4, 2.1

    Xs = Xr + dx
    Ys = Yr + dy
    r2 = (Xs - math.pi) ** 2 + (Ys - math.pi) ** 2
    sigma = 0.20 * 2 * math.pi
    amp = 1.0 * torch.exp(-r2 / (2 * sigma * sigma))
    phase = (c1 * torch.sin(1 * Xs + p1) * torch.cos(1 * Ys - p1) +
             c2 * torch.sin(2 * Xs + p2) * torch.cos(2 * Ys - p2) +
             c3 * torch.sin(3 * Xs + p3) * torch.cos(3 * Ys - p3))
    psi0 = amp * torch.exp(1j * phase)

    psi = psi0
    for _ in range(n_steps):
        psi = gp_step(psi, dt, g, k2)
    return psi0, psi


class ISNBlock(nn.Module):
    def __init__(self, hidden_dim=128, k=K_ISO, sigma_kernel=1.2):
        super().__init__()
        self.k = k
        self.sigma_kernel = sigma_kernel
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.angular_weight = nn.Linear(4, 1, bias=False)

    def forward(self, h, coords, e1, e2, neighbor_idx, sigma, z_depth,
                ablate_phase=False):
        B, N, C = h.shape
        K = neighbor_idx.shape[-1]
        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1,
            idx_flat.unsqueeze(-1).expand(-1, -1, 2)).view(B, N, K, 2)
        h_j = torch.gather(h, 1,
            idx_flat.unsqueeze(-1).expand(-1, -1, C)).view(B, N, K, C)
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
        coords_i = coords.unsqueeze(2).expand(-1, -1, K, -1)
        sigma_i = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
        z_depth_i = z_depth.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)

        delta = coords_j - coords_i
        dx = (delta * e1.unsqueeze(2)).sum(-1, keepdim=True)
        dy = (delta * e2.unsqueeze(2)).sum(-1, keepdim=True)
        mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8) / (sigma_i + 1e-8)
        w_radial = torch.exp(-0.5 * (mag / self.sigma_kernel) ** 2)

        phase = torch.atan2(dy, dx)
        cos2 = torch.cos(2 * phase)
        sin2 = torch.sin(2 * phase)
        cos4 = torch.cos(4 * phase)
        sin4 = torch.sin(4 * phase)

        if ablate_phase:
            cos2 = torch.ones_like(cos2)
            sin2 = torch.zeros_like(sin2)
            cos4 = torch.ones_like(cos4)
            sin4 = torch.zeros_like(sin4)

        angular_feats = torch.cat([cos2, sin2, cos4, sin4], dim=-1)
        angular_gate = torch.sigmoid(self.angular_weight(angular_feats))
        total_weight = angular_gate * w_radial

        edge_input = torch.cat(
            [h_i, h_j, mag, cos2, sin2, z_depth_i], dim=-1)
        messages = self.edge_mlp(edge_input) * total_weight
        agg = messages.sum(2) / (total_weight.sum(2) + 1e-8)

        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h + self.res_scale * self.norm(h_new)


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
        return torch.stack([
            torch.fft.ifft2(1j * KX * f_hat).real,
            torch.fft.ifft2(1j * KY * f_hat).real,
        ], dim=-1)

    def forward(self, psi_r, psi_i, coords, neighbor_idx, N,
                ablate_phase=False):
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
        e1 = gate * grad_norm + (1 - gate) * ref_vec
        e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
        e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)

        gr_lx = (grad_psi_r * e1).sum(-1, keepdim=True)
        gr_ly = (grad_psi_r * e2).sum(-1, keepdim=True)
        gi_lx = (grad_psi_i * e1).sum(-1, keepdim=True)
        gi_ly = (grad_psi_i * e2).sum(-1, keepdim=True)

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1,
            idx_flat.unsqueeze(-1).expand(-1, -1, 2)).view(B, N * N, self.k, 2)
        dists = (coords.unsqueeze(2) - coords_j).norm(dim=-1)
        sigma = dists.mean(-1).clamp(min=1e-6)
        z_val = self.alpha * math.log(self.R_phys / self.sigma_0 + 1e-8)
        z_depth = torch.full_like(k, z_val)

        x = torch.cat([k.unsqueeze(-1), grad_k_mag.unsqueeze(-1),
                       gr_lx, gr_ly, gi_lx, gi_ly,
                       z_depth.unsqueeze(-1)], dim=-1)

        h = self.input_mlp(x)
        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth,
                      ablate_phase=ablate_phase)

        delta = self.output_linear(h)
        return delta[..., 0], delta[..., 1]


def interior_slice(N):
    d = max(1, N // 16)
    c = 2 * d
    return c, N - c


@torch.no_grad()
def rel_rmse(pred_r, pred_i, t_r, t_i, N):
    c, e = interior_slice(N)
    B = pred_r.shape[0]
    pr = pred_r.view(B, N, N)[:, c:e, c:e]
    pi = pred_i.view(B, N, N)[:, c:e, c:e]
    tr = t_r.view(B, N, N)[:, c:e, c:e]
    ti = t_i.view(B, N, N)[:, c:e, c:e]
    num = torch.sqrt(((pr - tr) ** 2).sum() + ((pi - ti) ** 2).sum())
    den = torch.sqrt((tr ** 2).sum() + (ti ** 2).sum()) + 1e-12
    return (num / den).item()


def mse_combined(pred_r, pred_i, t_r, t_i, N):
    c, e = interior_slice(N)
    B = pred_r.shape[0]
    pr = pred_r.view(B, N, N)[:, c:e, c:e]
    pi = pred_i.view(B, N, N)[:, c:e, c:e]
    tr = t_r.view(B, N, N)[:, c:e, c:e]
    ti = t_i.view(B, N, N)[:, c:e, c:e]
    return 0.5 * (F.mse_loss(pr, tr).item() + F.mse_loss(pi, ti).item())


@torch.no_grad()
def run_single_step(model, device, N, seed, g=-1.0, angle=0.0,
                    ablate_phase=False):
    psi_t, psi_tp = generate_gp_pair(N, device, angle_deg=angle,
                                     seed=seed, g=g, n_steps=1)
    psi_r = psi_t.real.flatten().unsqueeze(0)
    psi_i = psi_t.imag.flatten().unsqueeze(0)
    t_r = psi_tp.real.flatten().unsqueeze(0)
    t_i = psi_tp.imag.flatten().unsqueeze(0)
    knn = get_cached_iso_knn(N, device)
    coords = model_coords(N, device)

    d_r, d_i = model(psi_r, psi_i, coords, knn, N,
                     ablate_phase=ablate_phase)
    pred_r = psi_r + d_r
    pred_i = psi_i + d_i

    rr = rel_rmse(pred_r, pred_i, t_r, t_i, N)
    mm = mse_combined(pred_r, pred_i, t_r, t_i, N)

    del psi_t, psi_tp, psi_r, psi_i, t_r, t_i
    del d_r, d_i, pred_r, pred_i, knn, coords
    return rr, mm


def eval_gi_ablation(model, device, N=64, n_samples=8, g=-1.0):
    print("\n" + "=" * 60)
    print("TEST 1 — gi=0 PHASE ABLATION")
    print("=" * 60)
    rrs_full, rrs_abl = [], []
    mse_full, mse_abl = [], []
    for i in range(n_samples):
        seed = 1000 + i
        rr, mm = run_single_step(model, device, N, seed, g=g,
                                  ablate_phase=False)
        rrs_full.append(rr); mse_full.append(mm)
        free_vram()
        rr, mm = run_single_step(model, device, N, seed, g=g,
                                  ablate_phase=True)
        rrs_abl.append(rr); mse_abl.append(mm)
        free_vram()

    rr_f = np.mean(rrs_full); rr_a = np.mean(rrs_abl)
    mse_f = np.mean(mse_full); mse_a = np.mean(mse_abl)
    print(f"  With phase:    RelRMSE={rr_f*100:.4f}%   MSE={mse_f:.2e}")
    print(f"  gi=0 ablated:  RelRMSE={rr_a*100:.4f}%   MSE={mse_a:.2e}")
    print(f"  MSE ratio (ablated/full): {mse_a/mse_f:.2f}x")
    print(f"  RelRMSE ratio:            {rr_a/rr_f:.2f}x")
    if mse_a / mse_f > 2.0:
        print(f"  -> Phase mechanism is LOAD-BEARING")
    else:
        print(f"  -> Phase mechanism is NOT load-bearing")


def eval_multi_ic(model, device, N=64, n_samples=16, g=-1.0):
    print("\n" + "=" * 60)
    print(f"TEST 2 — MULTI-IC GENERALIZATION (g={g})")
    print("=" * 60)
    rrs = []
    for i in range(n_samples):
        seed = 50000 + i
        rr, _ = run_single_step(model, device, N, seed, g=g)
        rrs.append(rr)
        free_vram()
    rrs = np.array(rrs)
    print(f"  mean RelRMSE = {rrs.mean()*100:.4f}%")
    print(f"  std          = {rrs.std()*100:.4f}%")
    print(f"  min / max    = {rrs.min()*100:.4f}% / {rrs.max()*100:.4f}%")


def eval_multi_g(model, device, N=64, g_values=(-0.5, -1.0, -2.0)):
    print("\n" + "=" * 60)
    print("TEST 3 — MULTI-g REGIME ROBUSTNESS")
    print("=" * 60)
    for g in g_values:
        rrs = []
        for i in range(8):
            rr, _ = run_single_step(model, device, N, 7000 + i, g=g)
            rrs.append(rr)
            free_vram()
        rrs = np.array(rrs)
        print(f"  g={g:+.1f}  RelRMSE={rrs.mean()*100:.4f}%  "
              f"std={rrs.std()*100:.4f}%")


def eval_rotation(model, device, N=64):
    print("\n" + "=" * 60)
    print(f"TEST 4 — ROTATION SWEEP (RelRMSE at {N}x{N})")
    print("=" * 60)
    rrs = []
    for angle in range(0, 360, 15):
        rr, _ = run_single_step(model, device, N, seed=42, angle=angle)
        rrs.append(rr)
        print(f"  {angle:3d}deg  RelRMSE={rr*100:.4f}%")
        free_vram()
    rrs = np.array(rrs)
    print(f"  mean={rrs.mean()*100:.4f}%  std={rrs.std()*100:.4f}%  "
          f"max={rrs.max()*100:.4f}%")


def eval_scale(model, device):
    print("\n" + "=" * 60)
    print("TEST 5 — SCALE SWEEP (RelRMSE, zero-shot)")
    print("=" * 60)
    for N in [16, 32, 64, 128, 256]:
        free_vram()
        try:
            rr, _ = run_single_step(model, device, N, seed=123)
            print(f"  {N}x{N}: RelRMSE={rr*100:.4f}%")
        except torch.cuda.OutOfMemoryError:
            print(f"  {N}x{N}: SKIPPED (VRAM limit)")
            free_vram()
    print("=" * 60)


def main():
    free_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    model = AlienXGP(hidden_dim=128, num_blocks=4).to(device)
    ckpt = torch.load("alienx_gp_v3_best.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"loaded alienx_gp_v3_best.pt  "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    eval_gi_ablation(model, device, N=64, n_samples=8, g=-1.0)
    free_vram()
    eval_multi_ic(model, device, N=64, n_samples=16, g=-1.0)
    free_vram()
    eval_multi_g(model, device, N=64, g_values=(-0.5, -1.0, -2.0))
    free_vram()
    eval_rotation(model, device, N=64)
    free_vram()
    eval_scale(model, device)

    print("\n" + "=" * 60)
    print("ALL EVALS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()