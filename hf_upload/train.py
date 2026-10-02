"""
AlienX (ISN) - 2D Release v2: Training + Evaluation Suite
=========================================================
Training config (exactly as run for the v2 results):
  - AdamW, lr 2e-3, weight decay 1e-4
  - Cosine annealing over 500 epochs (eta_min 1e-6)
  - Resolutions {16, 32, 64, 128}, 16 samples each (rotations random in [0, 360))
  - Interior MSE with boundary cropping scaled by dilation
  - Dynamic batch sampler: {16: 8, 32: 4, 64: 2, 128: 1}

Evaluation:
  - Zero-shot scale invariance (16 -> 256, 256 is zero-shot)
  - Rotation equivariance at {0, 45, 90, 180, 270} degrees
  - Arbitrary continuous angles {13, 27, 77, 123, 199}
  - Full 1-degree continuous rotation sweep + plot at 64x64

Reproduces the v2 numbers in RESULTS.md on a single T4 GPU.

Eric Yaka (Elbalor / The Digital Necromancer)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from model import AlienXOperator, get_cached_iso_knn
from data import generate_darcy_sample_gpu


# ==============================================================================
# GPU DATASET (pre-generated on GPU)
# ==============================================================================

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
    """Batches samples of the same resolution together (variable batch sizes)."""

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
                batch = indices[i:i + batch_size]
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


def compute_interior_loss(pred, target, grid_size):
    """Interior MSE with boundary crop scaled by dilation (Fix 1)."""
    B = pred.shape[0]
    p_grid = pred.view(B, grid_size, grid_size)
    t_grid = target.view(B, grid_size, grid_size)
    dilation = max(1, grid_size // 16)
    crop = 2 * dilation
    p_interior = p_grid[:, crop:-crop, crop:-crop]
    t_interior = t_grid[:, crop:-crop, crop:-crop]
    return F.mse_loss(p_interior, t_interior)


# ==============================================================================
# TRAINING
# ==============================================================================

def train(epochs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlienXOperator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    resolutions = [16, 32, 64, 128]
    dataset = DarcyDatasetGPU(resolutions, samples_per_res=16, device='cuda')

    batch_size_map = {16: 8, 32: 4, 64: 2, 128: 1}
    batch_sampler = DynamicBatchSampler(dataset, batch_size_map)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn_gpu, num_workers=0)

    best_loss = float('inf')

    for epoch in range(epochs):
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
                print(f"WARNING: NaN detected at epoch {epoch}. Reverting to best model.")
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
    return model


# ==============================================================================
# EVALUATION
# ==============================================================================

def _evaluate_at(model, device, res, angle, seed=123):
    """Single-sample interior L1 error at a resolution/angle. Returns (err, tensors)."""
    coords, k, grad_mag, grad_vec, p = generate_darcy_sample_gpu(res, device, angle_deg=angle, seed=seed)
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
    err = F.l1_loss(pred.view(1, res, res)[:, crop:-crop, crop:-crop],
                    p.view(1, res, res)[:, crop:-crop, crop:-crop]).item()
    return err


def evaluate_scale(model, device):
    print("\nZero-Shot Scale Invariance (Interior L1):")
    for res in [16, 32, 64, 128, 256]:
        err = _evaluate_at(model, device, res, 0)
        print(f"{res}x{res}: {err:.6f}")


def evaluate_rotation(model, device):
    print("\nRotation Equivariance (Interior L1):")
    for res in [16, 32, 64, 128, 256]:
        print(f"\nResolution {res}x{res}:")
        for angle in [0, 45, 90, 180, 270]:
            err = _evaluate_at(model, device, res, angle)
            print(f"  {angle}: {err:.6f}")


def evaluate_arbitrary(model, device):
    angles = [13.0, 27.0, 77.0, 123.0, 199.0]
    print("\nArbitrary Continuous Angle Equivariance Test (Interior L1):")
    for res in [16, 32, 64, 128, 256]:
        print(f"\n--- Resolution {res}x{res} ---")
        baseline_err = _evaluate_at(model, device, res, 0)
        print(f"  0.0 (Baseline) : {baseline_err:.6f}")

        for angle in angles:
            err = _evaluate_at(model, device, res, angle)
            delta = abs(err - baseline_err)
            print(f"  {angle:5.1f}          : {err:.6f}  (Delta vs 0 = {delta:.6f})")


def plot_continuous_rotation(model, device, res=64):
    import matplotlib.pyplot as plt

    angles = list(range(0, 360, 1))
    errors = []
    for angle in angles:
        errors.append(_evaluate_at(model, device, res, angle))

    plt.figure(figsize=(10, 5), facecolor='#0a0f1c')
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
