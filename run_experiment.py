import torch
import torch.nn as nn
import numpy as np
from model import AlienXOperator, get_deterministic_grid_knn
from data import generate_darcy_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlienXOperator(hidden_dim=384, L=4).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
epochs = 1000
warmup_epochs = 50
resolutions = [16, 32, 64, 128]

# --- TRAINING LOOP ---
print("🔥 Starting AlienX Training...")
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Dynamic LR Warmup + Cosine Decay
    if epoch < warmup_epochs:
        lr = 2e-3 * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    total_loss = 0
    # 48 samples per epoch, gradient accumulation over 4 steps
    for step in range(12): 
        loss_step = 0
        for _ in range(4):
            res = np.random.choice(resolutions)
            angle = np.random.uniform(0, 360) if np.random.rand() < 0.3 else 0.0
            
            coords, k, mag, vec, p = generate_darcy_sample(res, angle_deg=angle)
            coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
            k_t = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            mag_t = torch.tensor(mag, dtype=torch.float32, device=device).unsqueeze(0)
            vec_t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
            
            knn_idx = get_deterministic_grid_knn(res, device).to(device)
            
            pred = model(k_t, mag_t, vec_t, coords_t, knn_idx)
            rel_l2 = torch.norm(pred - p_t, dim=-1) / (torch.norm(p_t, dim=-1) + 1e-8)
            loss_step += rel_l2.mean()
            
        (loss_step / 4).backward()
        total_loss += loss_step.item() / 4
        
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/12:.5f} | LR: {lr:.2e}")

# --- EVALUATION (Generates Section 5 Tables) ---
print("\n📊 Generating Evaluation Tables...")
model.eval()
with torch.no_grad():
    # Table 1: Scale Invariance (0 degrees)
    print("\n--- Scale Invariance (Zero-Shot Transfer) ---")
    for res in [16, 32, 64, 128, 256]:
        coords, k, mag, vec, p = generate_darcy_sample(res, angle_deg=0.0)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
        k_t = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        mag_t = torch.tensor(mag, dtype=torch.float32, device=device).unsqueeze(0)
        vec_t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
        
        knn_idx = get_deterministic_grid_knn(res, device).to(device)
        pred = model(k_t, mag_t, vec_t, coords_t, knn_idx)
        rel_l2 = torch.norm(pred - p_t, dim=-1) / (torch.norm(p_t, dim=-1) + 1e-8)
        print(f"{res}x{res} | Rel L2 Error: {rel_l2.mean().item():.6f}")

    # Table 2: Rotation Equivariance
    print("\n--- Rotation Equivariance (64x64) ---")
    for angle in [0, 45, 90, 180, 270]:
        coords, k, mag, vec, p = generate_darcy_sample(64, angle_deg=angle)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
        k_t = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        mag_t = torch.tensor(mag, dtype=torch.float32, device=device).unsqueeze(0)
        vec_t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
        
        knn_idx = get_deterministic_grid_knn(64, device).to(device)
        pred = model(k_t, mag_t, vec_t, coords_t, knn_idx)
        rel_l2 = torch.norm(pred - p_t, dim=-1) / (torch.norm(p_t, dim=-1) + 1e-8)
        print(f"{angle:3d}° | Rel L2 Error: {rel_l2.mean().item():.6f}")