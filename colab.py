import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from google.colab import drive

# =============================================================================
# MOUNT GOOGLE DRIVE
# =============================================================================
drive.mount('/content/drive')

# =============================================================================
# MODEL
# =============================================================================
_KNN_CACHE = {}

def get_deterministic_grid_knn(grid_size, device, k=8, dilation=1):
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
    key = (grid_size, str(device), k, dilation)
    if key not in _KNN_CACHE:
        _KNN_CACHE[key] = get_deterministic_grid_knn(grid_size, device, k, dilation)
    return _KNN_CACHE[key]

def compute_local_frames(grad_k_vec, coords, neighbor_idx, k_field, eps=1e-4):
    B, N, _ = coords.shape
    K = neighbor_idx.shape[-1]

    grad_norm = torch.norm(grad_k_vec, dim=-1, keepdim=True)
    e1_grad = grad_k_vec / (grad_norm + 1e-8)

    idx_flat = neighbor_idx.reshape(B, -1)
    k_j = torch.gather(k_field, 1, idx_flat).view(B, N, K)
    k_i = k_field.unsqueeze(2)
    delta_k = torch.abs(k_j - k_i)
    weights = F.softmax(delta_k * 10.0, dim=2)

    coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B, N, K, 2)
    delta_x = coords_j - coords.unsqueeze(2)
    dist = torch.norm(delta_x, dim=-1, keepdim=True) + 1e-8
    unit_dir = delta_x / dist

    ref_vec = torch.sum(weights.unsqueeze(-1) * unit_dir, dim=2)
    ref_norm = torch.norm(ref_vec, dim=-1, keepdim=True) + 1e-8
    e1_fallback = ref_vec / ref_norm

    alpha = torch.sigmoid((grad_norm - eps) / (eps * 0.5))
    e1 = alpha * e1_grad + (1.0 - alpha) * e1_fallback
    e1 = e1 / (torch.norm(e1, dim=-1, keepdim=True) + 1e-8)

    e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)
    return e1, e2

class ISNBlock(nn.Module):
    def __init__(self, hidden_dim=384):
        super().__init__()
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
        B, N, _ = coords.shape
        grid_size = int(math.sqrt(N))
        device = coords.device

        neighbor_idx = get_cached_knn(grid_size, device, k=8, dilation=1)
        neighbor_idx = neighbor_idx.repeat(B, 1, 1)

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B, N, -1, 2)
        coords_i = coords.unsqueeze(2)
        distances = torch.norm(coords_i - coords_j, dim=-1)
        sigma = distances.mean(dim=-1, keepdim=True).clamp(min=1e-6)

        z_depth = self.alpha * torch.log(sigma / self.sigma_0 + 1e-8)

        e1, e2 = compute_local_frames(grad_k_vec, coords, neighbor_idx, k)

        grad_k_mag_norm = grad_k_mag * sigma.squeeze(-1)
        x_in = torch.cat([k.unsqueeze(-1), grad_k_mag_norm.unsqueeze(-1), z_depth], dim=-1)
        h = self.input_proj(x_in)

        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth)

        p_pred = self.output_proj(h).squeeze(-1)
        return p_pred

# =============================================================================
# DATA
# =============================================================================
def generate_darcy_sample(resolution=64, seed=None, angle_deg=0.0):
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

    k = 1.0 + 0.3 * np.sin(2*np.pi*X_r + 1.0) * np.cos(3*np.pi*Y_r - 0.5)
    k += 0.2 * np.sin(4*np.pi*X_r*Y_r)
    k = np.clip(k, 0.5, 2.0)

    p = np.sin(2*np.pi*X_r) * np.cos(2*np.pi*Y_r) + 0.5*np.sin(5*np.pi*X_r+1.3)*np.cos(4*np.pi*Y_r-0.7)

    dk_dX = (0.3 * (2*np.pi) * np.cos(2*np.pi*X_r + 1.0) * np.cos(3*np.pi*Y_r - 0.5) +
             0.2 * (4*np.pi*Y_r) * np.cos(4*np.pi*X_r*Y_r))
    dk_dY = (-0.3 * (3*np.pi) * np.sin(2*np.pi*X_r + 1.0) * np.sin(3*np.pi*Y_r - 0.5) +
             0.2 * (4*np.pi*X_r) * np.cos(4*np.pi*X_r*Y_r))

    grad_k_vec = np.stack([dk_dX.flatten(), dk_dY.flatten()], axis=-1)
    grad_k_mag = np.sqrt((grad_k_vec**2).sum(-1)).flatten()

    return coords_rot, k.flatten(), grad_k_mag, grad_k_vec, p.flatten()

# =============================================================================
# TRAINING
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlienXOperator(hidden_dim=384, L=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

epochs = 1000
warmup_epochs = 50
resolutions = [16, 32, 64, 128]

print("🔥 Starting AlienX Training on", device)

for epoch in range(1, epochs + 1):
    model.train()

    if epoch < warmup_epochs:
        lr = 2e-3 * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    total_loss = 0.0
    for step in range(12):
        optimizer.zero_grad()
        loss_step = 0.0
        for _ in range(4):
            res = np.random.choice(resolutions)
            angle = np.random.uniform(0, 360) if np.random.rand() < 0.3 else 0.0

            coords, k, mag, vec, p = generate_darcy_sample(res, angle_deg=angle)
            coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
            k_t = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            mag_t = torch.tensor(mag, dtype=torch.float32, device=device).unsqueeze(0)
            vec_t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)

            pred = model(coords_t, k_t, mag_t, vec_t)
            rel_l2 = torch.norm(pred - p_t, dim=-1) / (torch.norm(p_t, dim=-1) + 1e-8)
            loss_step += rel_l2.mean()

        (loss_step / 4).backward()
        optimizer.step()
        total_loss += loss_step.item() / 4

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/12:.5f} | LR: {lr:.2e}")

# =============================================================================
# SAVE MODEL TO GOOGLE DRIVE
# =============================================================================
save_path = '/content/drive/MyDrive/alienx_8nn.pt'
torch.save(model.state_dict(), save_path)
print(f"💾 Model saved to {save_path}")

# =============================================================================
# QUICK EVALUATION
# =============================================================================
model.eval()
print("\n📊 Quick Evaluation")

with torch.no_grad():
    for res in [16, 32, 64, 128, 256]:
        coords, k, mag, vec, p = generate_darcy_sample(res, seed=200, angle_deg=0.0)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
        k_t = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        mag_t = torch.tensor(mag, dtype=torch.float32, device=device).unsqueeze(0)
        vec_t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)

        pred = model(coords_t, k_t, mag_t, vec_t)
        rel_l2 = torch.norm(pred - p_t, dim=-1) / (torch.norm(p_t, dim=-1) + 1e-8)
        print(f"{res}x{res} | Rel L2 Error: {rel_l2.mean().item():.6f}")