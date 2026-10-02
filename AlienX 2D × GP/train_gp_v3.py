"""
AlienX 2D × GP — Training v3 (tight schedule + Dark Necromancer Graph)
========================================================================

Schedule:
  Epochs 0-40:    K=1   (learn single step)
  Epochs 40-100:  K=3   (learn short rollout)
  Epochs 100-200: K=5   (learn long rollout)
  Warmup: 20 epochs

Saves best checkpoint to Drive on every improvement.
Produces Dark Necromancer Graph + dark field comparison image at the end.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from google.colab import drive

# ═══════════════════════════════════════════════════════════════════════
# Setup: Drive + paths
# ═══════════════════════════════════════════════════════════════════════
drive.mount("/content/drive", force_remount=False)
SAVE_DIR = "/content/drive/MyDrive/AlienX_GP"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"save dir: {SAVE_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# Stencil
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


# ═══════════════════════════════════════════════════════════════════════
# GP solver + IC generator
# ═══════════════════════════════════════════════════════════════════════
def gp_step(psi, dt, g, k2):
    lin_half = torch.exp(-0.5j * dt * 0.5 * k2)
    psi_hat = torch.fft.fft2(psi) * lin_half
    psi = torch.fft.ifft2(psi_hat)
    dens = psi.real ** 2 + psi.imag ** 2
    psi = psi * torch.exp(-1j * g * dens * dt)
    psi_hat = torch.fft.fft2(psi) * lin_half
    return torch.fft.ifft2(psi_hat)


def make_ic(N, device, seed, angle_deg=0.0):
    g_ic = torch.Generator(device=device).manual_seed(seed)
    theta = angle_deg * math.pi / 180.0
    x = torch.linspace(0, 2 * math.pi, N + 1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="ij")
    c, s = math.cos(-theta), math.sin(-theta)
    Xr = c * (X - math.pi) - s * (Y - math.pi) + math.pi
    Yr = s * (X - math.pi) + c * (Y - math.pi) + math.pi

    dx = 0.25 * (torch.rand(1, generator=g_ic, device=device).item() - 0.5)
    dy = 0.25 * (torch.rand(1, generator=g_ic, device=device).item() - 0.5)
    c1 = 0.4 * (0.7 + 0.6 * torch.rand(1, generator=g_ic, device=device).item())
    c2 = 0.4 * (0.7 + 0.6 * torch.rand(1, generator=g_ic, device=device).item())
    c3 = 0.4 * (0.7 + 0.6 * torch.rand(1, generator=g_ic, device=device).item())
    p1 = 2.0 * math.pi * torch.rand(1, generator=g_ic, device=device).item()
    p2 = 2.0 * math.pi * torch.rand(1, generator=g_ic, device=device).item()
    p3 = 2.0 * math.pi * torch.rand(1, generator=g_ic, device=device).item()

    Xs = Xr + dx
    Ys = Yr + dy
    r2 = (Xs - math.pi) ** 2 + (Ys - math.pi) ** 2
    sigma = 0.20 * 2 * math.pi
    amp = 1.0 * torch.exp(-r2 / (2 * sigma * sigma))
    phase = (c1 * torch.sin(1 * Xs + p1) * torch.cos(1 * Ys - p1) +
             c2 * torch.sin(2 * Xs + p2) * torch.cos(2 * Ys - p2) +
             c3 * torch.sin(3 * Xs + p3) * torch.cos(3 * Ys - p3))
    return amp * torch.exp(1j * phase)


def make_trajectory(N, device, seed, angle_deg, g, dt, K):
    kx = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=device)
    ky = 2 * math.pi * torch.fft.fftfreq(N, d=1.0 / N, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2

    psi = make_ic(N, device, seed, angle_deg)
    frames = [psi]
    for _ in range(K):
        psi = gp_step(psi, dt, g, k2)
        frames.append(psi)
    return torch.stack(frames)


def build_dataset(resolutions, n_per_res, device, g, dt, K, seed_base):
    data = []
    for res in resolutions:
        for i in range(n_per_res):
            seed = seed_base + res * 100000 + i
            angle = float(np.random.uniform(0, 360))
            traj = make_trajectory(res, device, seed, angle, g, dt, K)
            data.append((traj, res))
    return data


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════
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

    def forward(self, h, coords, e1, e2, neighbor_idx, sigma, z_depth):
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
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth)

        delta = self.output_linear(h)
        return delta[..., 0], delta[..., 1]


# ═══════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════
def interior_slice(N):
    d = max(1, N // 16)
    c = 2 * d
    return c, N - c


def rel_rmse_loss(pred_r, pred_i, t_r, t_i, N):
    c, e = interior_slice(N)
    B = pred_r.shape[0]
    pr = pred_r.view(B, N, N)[:, c:e, c:e]
    pi = pred_i.view(B, N, N)[:, c:e, c:e]
    tr = t_r.view(B, N, N)[:, c:e, c:e]
    ti = t_i.view(B, N, N)[:, c:e, c:e]
    num = ((pr - tr) ** 2).sum() + ((pi - ti) ** 2).sum()
    den = (tr ** 2).sum() + (ti ** 2).sum() + 1e-12
    return num / den


def mass_loss(pred_r, pred_i, t_r, t_i):
    m_pred = (pred_r ** 2).sum() + (pred_i ** 2).sum()
    m_true = (t_r ** 2).sum() + (t_i ** 2).sum() + 1e-12
    return ((m_pred - m_true) / m_true) ** 2


def spectral_loss(pred_r, pred_i, t_r, t_i, N):
    pr = torch.fft.fft2(pred_r.view(-1, N, N))
    pi = torch.fft.fft2(pred_i.view(-1, N, N))
    tr = torch.fft.fft2(t_r.view(-1, N, N))
    ti = torch.fft.fft2(t_i.view(-1, N, N))
    E_pred = (pr.abs() ** 2 + pi.abs() ** 2).sum(dim=0).sum(dim=0)
    E_true = (tr.abs() ** 2 + ti.abs() ** 2).sum(dim=0).sum(dim=0)
    return ((E_pred - E_true) ** 2 / (E_true.sum() ** 2 + 1e-12)).sum()


def rollout_loss(model, traj, K, N, device, weights, detach_intermediate=True):
    coords = model_coords(N, device)
    knn = get_cached_iso_knn(N, device)

    psi = traj[0]
    psi_r = psi.real.flatten().unsqueeze(0)
    psi_i = psi.imag.flatten().unsqueeze(0)

    total = 0.0
    for step in range(1, K + 1):
        d_r, d_i = model(psi_r, psi_i, coords, knn, N)
        pred_r = psi_r + d_r
        pred_i = psi_i + d_i

        tgt = traj[step]
        t_r = tgt.real.flatten().unsqueeze(0)
        t_i = tgt.imag.flatten().unsqueeze(0)

        l_rr = rel_rmse_loss(pred_r, pred_i, t_r, t_i, N)
        l_m = mass_loss(pred_r, pred_i, t_r, t_i)
        l_s = spectral_loss(pred_r, pred_i, t_r, t_i, N)

        step_loss = (weights["rr"] * l_rr +
                     weights["mass"] * l_m +
                     weights["spec"] * l_s)
        total = total + step_loss

        if detach_intermediate and step < K:
            psi_r = pred_r.detach()
            psi_i = pred_i.detach()
        else:
            psi_r = pred_r
            psi_i = pred_i

    return total / K


# ═══════════════════════════════════════════════════════════════════════
# LR / K schedules
# ═══════════════════════════════════════════════════════════════════════
def get_lr(epoch, warmup, total, peak_lr, min_lr):
    if epoch < warmup:
        return peak_lr * (epoch + 1) / warmup
    r = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (peak_lr - min_lr)


def get_K(epoch, schedule):
    k = 1
    for thresh, kk in schedule:
        if epoch >= thresh:
            k = kk
    return k


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════
def train(args):
    free_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\nbuilding dataset...")
    t0 = time.time()
    resolutions = [16, 32, 64]
    K_max = max(k for _, k in args.k_schedule)
    train_data = build_dataset(
        resolutions, args.n_train, device, args.g, args.dt, K_max,
        seed_base=0,
    )
    val_data = build_dataset(
        resolutions, args.n_val, device, args.g, args.dt, K_max,
        seed_base=10_000_000,
    )
    print(f"  train: {len(train_data)} samples  "
          f"val: {len(val_data)} samples  ({time.time()-t0:.1f}s)")

    model = AlienXGP(hidden_dim=args.hidden_dim,
                     num_blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.wd)

    weights = {"rr": 1.0, "mass": 0.1, "spec": 0.05}

    history = {"epoch": [], "K": [], "train": [], "val": [], "lr": []}

    print(f"\ntraining {args.epochs} epochs, warmup={args.warmup}")
    print(f"K-schedule: {args.k_schedule}")
    print("=" * 70)

    best_val = float("inf")
    t_start = time.time()

    for epoch in range(args.epochs):
        lr = get_lr(epoch, args.warmup, args.epochs, args.lr, args.lr_min)
        for g in opt.param_groups:
            g["lr"] = lr
        K = get_K(epoch, args.k_schedule)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        np.random.shuffle(train_data)
        for traj, res in train_data:
            loss = rollout_loss(model, traj, K, res, device, weights)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

            if torch.isnan(loss):
                print(f"NaN at epoch {epoch}. Reverting to best.")
                best_path = os.path.join(SAVE_DIR, "alienx_gp_v3_best.pt")
                if os.path.exists(best_path):
                    model.load_state_dict(torch.load(
                        best_path, map_location=device))
                return history

        avg_train = epoch_loss / max(n_batches, 1)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_n = 0
                for traj, res in val_data[:24]:
                    l = rollout_loss(model, traj, 1, res, device, weights,
                                     detach_intermediate=False)
                    val_loss += l.item()
                    val_n += 1
                val_loss /= max(val_n, 1)

            history["epoch"].append(epoch)
            history["K"].append(K)
            history["train"].append(avg_train)
            history["val"].append(val_loss)
            history["lr"].append(lr)

            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR, "alienx_gp_v3_best.pt"))
                marker = " ★"

            print(f"epoch {epoch:4d}  K={K}  train={avg_train:.6f}  "
                  f"val={val_loss:.6f}  lr={lr:.1e}  "
                  f"({time.time()-t_start:.0f}s){marker}")

    torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, "alienx_gp_v3_final.pt"))
    print(f"\nbest val: {best_val:.6f}")
    print(f"total time: {(time.time()-t_start)/60:.1f} min")

    # Save history
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


# ═══════════════════════════════════════════════════════════════════════
# Dark Necromancer Graph
# ═══════════════════════════════════════════════════════════════════════
def dark_necromancer_graph(history, save_dir):
    BG = "#0a0f1c"
    FG = "#d8e0f0"
    GRID = "#1f2d40"
    ACCENT_1 = "#7fd8a8"   # train
    ACCENT_2 = "#f4a261"   # val
    ACCENT_3 = "#6b93c7"   # lr
    ACCENT_4 = "#c1121f"   # K markers

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    for ax in axes.ravel():
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, alpha=0.4)

    epochs = history["epoch"]
    train = history["train"]
    val = history["val"]
    lr = history["lr"]
    K_list = history["K"]

    # Panel 1: loss curves
    ax = axes[0, 0]
    ax.semilogy(epochs, train, color=ACCENT_1, linewidth=2, label="train")
    ax.semilogy(epochs, val, color=ACCENT_2, linewidth=2, label="val")
    for i in range(1, len(epochs)):
        if K_list[i] != K_list[i - 1]:
            ax.axvline(epochs[i], color=ACCENT_4, linestyle="--",
                       alpha=0.5, linewidth=1)
            ax.text(epochs[i], max(val) * 0.5, f" K={K_list[i]}",
                    color=ACCENT_4, fontsize=10)
    ax.set_xlabel("epoch", color=FG)
    ax.set_ylabel("loss (log)", color=FG)
    ax.set_title("Loss curves", color=FG, fontsize=13)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)

    # Panel 2: RelRMSE from val loss
    ax = axes[0, 1]
    val_relrmse = [100 * math.sqrt(v) for v in val]
    ax.plot(epochs, val_relrmse, color=ACCENT_2, linewidth=2,
            marker="o", markersize=3)
    ax.axhline(1.0, color="#8b95a5", linestyle=":", alpha=0.6,
               label="dFNO+1 <1%")
    ax.set_xlabel("epoch", color=FG)
    ax.set_ylabel("RelRMSE (%)", color=FG)
    ax.set_title("Val RelRMSE (approx from loss)", color=FG, fontsize=13)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)

    # Panel 3: LR
    ax = axes[1, 0]
    ax.plot(epochs, lr, color=ACCENT_3, linewidth=2)
    ax.set_xlabel("epoch", color=FG)
    ax.set_ylabel("learning rate", color=FG)
    ax.set_title("LR schedule", color=FG, fontsize=13)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Panel 4: K schedule
    ax = axes[1, 1]
    ax.step(epochs, K_list, color=ACCENT_4, linewidth=2, where="post")
    ax.set_xlabel("epoch", color=FG)
    ax.set_ylabel("rollout K", color=FG)
    ax.set_title("K-unroll curriculum", color=FG, fontsize=13)
    ax.set_yticks(sorted(set(K_list)))

    fig.suptitle("Dark Necromancer Graph — AlienX 2D × Gross-Pitaevskii",
                 color=FG, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(save_dir, "dark_necromancer_graph_gp.png")
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"\n[plot] saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════
# Dark field comparison image
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def dark_field_image(model, device, save_dir, N=64, seed=2024, g=-1.0, dt=2e-3):
    """Compare true GP trajectory vs AlienX rollout on one IC, dark themed."""
    model.eval()
    knn = get_cached_iso_knn(N, device)
    coords = model_coords(N, device)

    # True trajectory
    true_traj = make_trajectory(N, device, seed, 0.0, g, dt, K=5)

    # Model rollout
    psi = true_traj[0]
    psi_r = psi.real.flatten().unsqueeze(0)
    psi_i = psi.imag.flatten().unsqueeze(0)
    pred_traj = [psi]
    for _ in range(5):
        d_r, d_i = model(psi_r, psi_i, coords, knn, N)
        psi_r = psi_r + d_r
        psi_i = psi_i + d_i
        pred_traj.append(torch.complex(psi_r.view(N, N),
                                        psi_i.view(N, N)))

    # Layout: 2 rows (true, pred), 6 cols (frames 0..5)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), facecolor="#0a0f1c")
    for ax in axes.ravel():
        ax.set_facecolor("#0a0f1c")
        ax.axis("off")

    # Density colormap — dark purple to gold
    colors = ["#0a0f1c", "#1a2b4c", "#2d4a7a", "#4a6fa5",
              "#6b93c7", "#f4a261", "#e76f51", "#c1121f", "#ffd166"]
    cmap = LinearSegmentedColormap.from_list("dark_necromancer", colors)

    for col in range(6):
        psi_t = true_traj[col]
        psi_p = pred_traj[col]
        dens_t = (psi_t.real ** 2 + psi_t.imag ** 2).cpu().numpy()
        dens_p = (psi_p.real ** 2 + psi_p.imag ** 2).cpu().numpy()

        vmax = max(dens_t.max(), dens_p.max())

        axes[0, col].imshow(dens_t, cmap=cmap, vmin=0, vmax=vmax)
        axes[0, col].set_title(f"t={col*dt:.3f}", color="#d8e0f0", fontsize=10)

        axes[1, col].imshow(dens_p, cmap=cmap, vmin=0, vmax=vmax)

    axes[0, 0].set_ylabel("truth", color="#7fd8a8", fontsize=12)
    axes[1, 0].set_ylabel("AlienX", color="#f4a261", fontsize=12)

    fig.suptitle("AlienX 2D × GP — rollout comparison (|ψ|²)",
                 color="#d8e0f0", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(save_dir, "dark_field_rollout_gp.png")
    fig.savefig(path, dpi=150, facecolor="#0a0f1c")
    plt.close(fig)
    print(f"[plot] saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
class Args:
    hidden_dim = 128
    blocks = 4
    g = -1.0
    dt = 2e-3
    seed = 0
    lr = 2e-3
    lr_min = 5e-6
    wd = 1e-4
    warmup = 20
    epochs = 200
    n_train = 64
    n_val = 12
    eval_every = 10
    k_schedule = [(0, 1), (40, 3), (100, 5)]


if __name__ == "__main__":
    args = Args()
    history = train(args)

    # Reload best for eval + plots
    free_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlienXGP(hidden_dim=args.hidden_dim,
                     num_blocks=args.blocks).to(device)
    best_path = os.path.join(SAVE_DIR, "alienx_gp_v3_best.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    print(f"\nloaded best: {best_path}")

    dark_necromancer_graph(history, SAVE_DIR)
    dark_field_image(model, device, SAVE_DIR, N=64, seed=2024, g=-1.0, dt=2e-3)

    print(f"\nall artifacts saved to: {SAVE_DIR}")
    for f in sorted(os.listdir(SAVE_DIR)):
        size = os.path.getsize(os.path.join(SAVE_DIR, f)) / 1024
        print(f"  {f}  ({size:.1f} KB)")