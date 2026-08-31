import torch
import torch.nn as nn
import numpy as np
from model import AlienXOperator
from data import generate_darcy_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlienXOperator(hidden_dim=384, L=4).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
epochs = 1000
warmup_epochs = 50
resolutions = [16, 32, 64, 128]

print("🔥 Starting AlienX Training...")
for epoch in range(1, epochs + 1):
    model.train()

    # Dynamic LR Warmup + Cosine Decay
    if epoch < warmup_epochs:
        lr = 2e-3 * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    total_loss = 0.0
    # 12 outer steps, each accumulating 4 samples = 48 samples/epoch
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

            pred = model(coords_t, k_t, mag_t, vec_t)   # correct call
            rel_l2 = torch.norm(pred - p_t, dim=-1) / (torch.norm(p_t, dim=-1) + 1e-8)
            loss_step += rel_l2.mean()

        (loss_step / 4).backward()
        optimizer.step()
        total_loss += loss_step.item() / 4

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/12:.5f} | LR: {lr:.2e}")

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
print("\n📊 Generating Evaluation Tables...")
model.eval()
with torch.no_grad():
    print("\n--- Scale Invariance (Zero-Shot Transfer) ---")
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

    print("\n--- Rotation Equivariance (64x64) ---")
    for angle in [0, 45, 90, 180, 270]:
        coords, k, mag, vec, p = generate_darcy_sample(64, seed=201, angle_deg=angle)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
        k_t = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        mag_t = torch.tensor(mag, dtype=torch.float32, device=device).unsqueeze(0)
        vec_t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)

        pred = model(coords_t, k_t, mag_t, vec_t)
        rel_l2 = torch.norm(pred - p_t, dim=-1) / (torch.norm(p_t, dim=-1) + 1e-8)
        print(f"{angle:3d}° | Rel L2 Error: {rel_l2.mean().item():.6f}")