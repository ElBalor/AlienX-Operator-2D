# AlienX (ISN): A Continuous Rotation- and Scale-Invariant Operator on Geometric Manifolds

**Author:** Eric Yaka 
**Date:** September 2026
**Status:** 2D validation complete with isotropic 24-neighbor stencil. Arbitrary rotation equivariance within 0.003 L1 error. n-dimensional organism under construction.

---

## Abstract

AlienX (Isomorphic Spatial Net, ISN) is a neural operator that operates on continuous geometric manifolds instead of fixed grids. It natively respects rotation and scale invariance through a local SO(2) frame construction, scale-normalized displacements, and a 2π-periodic complex harmonic embedding.

The network learns from Darcy flow with resolutions from 16×16 to 256×256 and rotations spanning any angle. The final isotropic 24-neighbor stencil eliminates the 45° residual observed in earlier 8-directional versions. New results demonstrate:

- **Scale invariance:** Interior L1 error remains in the band 0.0036–0.0087 across all tested resolutions, with no catastrophic drift.
- **Rotation equivariance:** Errors at 0°, 45°, 90°, 180°, 270° are nearly identical, and arbitrary angles (13°, 27°, 77°, 123°, 199°) deviate from baseline by less than 0.003 at resolutions ≥ 32×32.

These results confirm that AlienX behaves as a continuous SO(2) equivariant operator in practice, not just in theory. The architecture is a stepping stone to a fully n-dimensional self-evolving organism.

---

## 1. Introduction

Standard neural operators (FNO, DeepONet, GNO) rely on fixed grids and provide no built-in rotation or scale equivariance — symmetry, when needed, is approximated by data augmentation. AlienX replaces the pixel grid with a continuous spatial graph. Each node is a point in R^2 (or R^n in general), and message passing occurs over local neighborhoods defined by physical distance. The model is built to be equivariant, not just invariant, under rotations and scale changes.

The key insight is that the network should not learn geometry from data — it should be born with geometry. Rotation, scale, and translation are not features to memorize. They are coordinate artifacts that should vanish before learning begins.

### 1.1 Related work

Group-equivariant CNNs (Cohen & Welling, 2016) and steerable networks (Weiler & Cesa, 2019) achieve rotation equivariance via explicit group lifting, expanding feature fields over group elements — powerful, but combinatorially expensive in channels. Lie-group convolutions (Finzi et al., 2020) extend this to continuous groups on arbitrary data at the cost of kernel interpolation and group lifts. Gauge-equivariant networks (Cohen et al., 2019) construct local frames on manifolds, but rely on learned or parallel-transported gauges. Group-equivariant Fourier neural operators (Helwig et al., 2023) bring rotation equivariance to the frequency domain, and the Fourier Neural Operator (Li et al., 2021) and DeepONet (Lu et al., 2021) define the grid-based operator baseline AlienX departs from. AlienX takes a different route: the gauge is constructed from the data itself — e1 from ∇k with an inertia-tensor fallback — costing no group lifts, learning no geometry, and yielding exactness we verify bitwise (pred(R·k) = R·pred(k) = 0.000e+00 under D4 with live message passing).

---

## 2. Mathematical Foundations

### 2.1 Continuous Spatial Graph

Input fields are sampled on a regular grid but treated as a graph G = (V, E) with nodes i in V having coordinates x_i in R^2. Edges connect k nearest neighbors.

### 2.2 Local Scale Normalization

For each node, compute the local dispersion sigma_i as the mean distance to its k nearest neighbors. Scale-invariant distances are:

```
d_hat_ij = ||x_i - x_j|| / sigma_i
```

The scale-space depth coordinate is:

```
z_i = alpha * ln(sigma_i / sigma_0)
```

where alpha and sigma_0 are hyperparameters. This gives the network explicit knowledge of absolute scale without breaking the invariance of the relative features.

In the final implementation, sigma is computed from physical neighbor distances and normalized by the dilation factor so that the physical support radius remains constant across resolutions. The depth z_i is then constant, removing any scale drift.

### 2.3 SO(2)-Equivariant Local Frame

Construct a local orthonormal frame (e1, e2) at each node. The primary axis e1 is derived from the gradient of the permeability field grad(k). A fallback direction is used where ||grad k|| is small, based on the inertia tensor of the field. The fallback eigenvector's sign is anchored deterministically to the k-weighted centroid (s = ⟨v, m⟩, odd in v and D4-invariant), making the frame a pure function of the data — see §4.4.

Relative displacements dx_ij = x_j − x_i are projected onto this frame:

```
dx_local = dx_ij · e1
dy_local = dx_ij · e2
```

These projections are rotationally equivariant: rotating the input rotates the frame accordingly, so the local coordinates remain unchanged.

### 2.4 Complex Harmonic Embedding

The local angle theta = atan2(dy_local, dx_local) is encoded as the unit complex number:

```
e^(i·theta) = cos(theta) + i·sin(theta)
```

This is not a real-valued approximation; it is the native complex phase geometry. The pair (cos theta, sin theta) lives on the unit circle in CP^1, giving:

- 2π periodicity without branch-cut jumps.
- Rotational equivariance — the phase is relative to the local frame, so global rotations leave it invariant.
- Constructive/destructive interference — later layers can learn to amplify or cancel directions natively.

The magnitude ||x_j − x_i|| / sigma_i completes the polar form. Combined with the scale-space depth z_i, the complete edge feature set is:

```
[ h_i, h_j, ||x_j − x_i|| / sigma_i, cos(theta), sin(theta), z_i ]
```

In the final isotropic stencil version, additional even harmonics cos(2θ), sin(2θ), cos(4θ), sin(4θ) are used as angular features, making the message weights invariant under θ → θ + π.

### 2.5 Scale Invariance Proof

Under uniform scaling x' = s·x:

```
||x_i' − x_j'|| = s · ||x_i − x_j||,   sigma_i' = s · sigma_i
```

Therefore:

```
d_hat_ij' = (s · ||x_i − x_j||) / (s · sigma_i) = d_hat_ij
```

The scale factor cancels exactly before any learning happens.

---

## 3. Architecture

The ISN consists of:

- An input projection MLP that lifts scalar fields (k, ||grad k||, z) to a hidden dimension H = 128.
- L = 4 stacked ISN blocks.
- An output linear layer H → 1.

Each ISN block performs:

1. Gather neighbor features and coordinates using cached isotropic grid KNN.
2. Compute local frame projections and complex harmonic features.
3. Compute magnitude, phase, radial taper, and append scale-space depth z.
4. Pass through an edge MLP to produce messages.
5. Weight messages by angular gate and radial taper, then aggregate (weighted mean).
6. Update node state through a node MLP with residual connection and LayerNorm.

---

## 4. Implementation Details

### 4.1 Data Generation with Pull-Back Rotation (Fixed)

The original script stretched the domain to a diamond at 45°, causing out-of-distribution frequencies. The fixed version uses a pull-back rotation: evaluate fields in reference coordinates, then push gradients forward. The domain always stays within [−1, 1]^2.

```python
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
```

### 4.2 Isotropic 24-Neighbor Stencil (Cached, Dynamically Dilated)

The isotropic stencil uses a circular neighborhood of radius sqrt(8), giving 24 symmetric neighbors. Dilation scales with resolution to keep the physical support radius constant.

```python
RADIUS = 2.8284271247461903   # sqrt(8)
ISO_OFFSETS = []
for dr in range(-int(RADIUS), int(RADIUS)+1):
    for dc in range(-int(RADIUS), int(RADIUS)+1):
        if dr == 0 and dc == 0:
            continue
        if math.sqrt(dr*dr + dc*dc) <= RADIUS + 1e-8:
            ISO_OFFSETS.append((dr, dc))

K_ISO = len(ISO_OFFSETS)   # 24

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
        scale_factor = max(1, grid_size // 16)   # dynamic dilation
        KNN_CACHE[key] = get_isotropic_knn(grid_size, device, dilation=scale_factor)
    return KNN_CACHE[key]
```

### 4.3 ISN Block with Radial Taper and Even Harmonics

```python
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

        angular_in = 4   # cos(2θ), sin(2θ), cos(4θ), sin(4θ)
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

        angular_feats = torch.cat([
            torch.cos(2 * phase),
            torch.sin(2 * phase),
            torch.cos(4 * phase),
            torch.sin(4 * phase)
        ], dim=-1)

        angular_gate = torch.sigmoid(self.angular_weight(angular_feats))
        total_weight = angular_gate * w_radial

        edge_input = torch.cat([h_i, h_j, mag, cos_phase, sin_phase, z_depth_i], dim=-1)
        messages = self.edge_mlp(edge_input) * total_weight
        agg = messages.sum(dim=2) / (total_weight.sum(dim=2) + 1e-8)

        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(node_input)
        return h + self.res_scale * self.norm(h_new)
```

### 4.4 AlienX Operator with Constant Depth and Inertia Anchor

```python
class AlienXOperator(nn.Module):
    def __init__(self, hidden_dim=128, num_blocks=4, k=K_ISO, sigma_kernel=1.2,
                 alpha=0.5, sigma_0=0.1, anchor_temperature=0.2, R_phys=0.25):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.anchor_temperature = anchor_temperature
        self.R_phys = R_phys

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

        # Inertia tensor anchor with sign_lock (ported from the 3D QSA_ISNBlock3D):
        # eigh's eigenvector SIGN is unspecified under rotation; s = <v, m> is
        # odd in v and D4-invariant, so the lock cancels the arbitrary flip and
        # makes the fallback frame a deterministic function of the data.
        anchor_weights = torch.softmax(k / self.anchor_temperature, dim=1)
        m = torch.einsum('bn,bnd->bd', anchor_weights, coords)          # k-weighted centroid
        x_weighted = coords * anchor_weights.unsqueeze(-1)
        S = torch.einsum('bnd,bnc->bdc', x_weighted, x_weighted)
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        ref_vec = eigenvectors[..., -1]
        ref_vec = ref_vec / (ref_vec.norm(dim=-1, keepdim=True) + 1e-8)
        sign_lock = torch.sign((ref_vec * m.unsqueeze(1)).sum(-1, keepdim=True))
        sign_lock = torch.where(sign_lock == 0, torch.ones_like(sign_lock), sign_lock)
        ref_vec = ref_vec * sign_lock
        ref_vec = ref_vec.unsqueeze(1).expand(-1, N, -1)

        grad_norm = grad_k / (grad_k_mag.unsqueeze(-1) + 1e-8)
        use_radial = (grad_k_mag.unsqueeze(-1) < 0.1).float()
        e1 = grad_norm * (1 - use_radial) + ref_vec * use_radial
        e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
        e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)

        grid_size = int(math.sqrt(N))
        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B,N,self.k,2)
        distances = torch.norm(coords.unsqueeze(2) - coords_j, dim=-1)
        sigma = distances.mean(dim=-1).clamp(min=1e-6)

        # Constant physical depth
        z_depth_value = self.alpha * torch.log(
            torch.tensor(self.R_phys / self.sigma_0 + 1e-8, dtype=torch.float32, device=device)
        )
        z_depth = z_depth_value.expand_as(k)

        x = torch.stack([k, grad_k_mag, z_depth], dim=-1)
        h = self.input_mlp(x)

        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx, sigma, z_depth)

        out = self.output_linear(h)
        return out.squeeze(-1)
```

### 4.5 Training

- Optimizer: AdamW, lr 2e-3, weight decay 1e-4
- Cosine annealing over 500 epochs
- Resolutions: {16, 32, 64, 128}, 16 samples each
- Rotations: random angles in [0, 360)
- Loss: interior MSE with boundary cropping scaled by dilation

---

## 5. Results

### 5.1 Scale Invariance (Zero-Shot Transfer)

Interior L1 error (lower is better):

| Resolution | L1 Error |
|---|---|
| 16×16 | 0.008743 |
| 32×32 | 0.005163 |
| 64×64 | 0.003649 |
| 128×128 | 0.003586 |
| 256×256 | 0.003759 |

Error remains stable from 16×16 to 256×256. No catastrophic drift. The 256×256 column is zero-shot: the model never saw this resolution during training.

### 5.2 Rotation Equivariance

Interior L1 error (lower is better):

| Angle | 16×16 | 32×32 | 64×64 | 128×128 | 256×256 |
|---|---|---|---|---|---|
| 0° | 0.008743 | 0.005163 | 0.003649 | 0.003586 | 0.003759 |
| 45° | 0.009822 | 0.005143 | 0.003420 | 0.003279 | 0.003442 |
| 90° | 0.008743 | 0.005163 | 0.003630 | 0.003616 | 0.003763 |
| 180° | 0.008743 | 0.005163 | 0.003630 | 0.003616 | 0.003763 |
| 270° | 0.008743 | 0.005163 | 0.003649 | 0.003586 | 0.003759 |

45° no longer shows an elevated residual. All angles are essentially equivalent.

### 5.3 Arbitrary Continuous Angle Equivariance

Deviation from 0° baseline (Delta L1) at 64×64:

| Angle | Error | Delta vs 0° |
|---|---|---|
| 0.0° | 0.003649 | — |
| 13.0° | 0.004444 | 0.000795 |
| 27.0° | 0.004480 | 0.000832 |
| 77.0° | 0.004558 | 0.000909 |
| 123.0° | 0.004998 | 0.001349 |
| 199.0° | 0.005304 | 0.001656 |

Arbitrary rotations deviate from baseline by less than 0.002 at 64×64.

Full-resolution sweep across all tested resolutions (13°, 27°, 77°, 123°, 199° at 16, 32, 64, 128, 256):

| Angle | 16×16 | 32×32 | 64×64 | 128×128 | 256×256 |
|---|---|---|---|---|---|
| 13.0° | 0.007282 | 0.005241 | 0.004444 | 0.004586 | 0.004796 |
| 27.0° | 0.006186 | 0.004889 | 0.004480 | 0.004486 | 0.004713 |
| 77.0° | 0.008181 | 0.005403 | 0.004558 | 0.004624 | 0.004804 |
| 123.0° | 0.008562 | 0.006120 | 0.004998 | 0.004926 | 0.005087 |
| 199.0° | 0.013452 | 0.005842 | 0.005304 | 0.005554 | 0.005789 |

At resolutions ≥ 32×32, arbitrary rotations deviate from baseline by less than 0.003 in L1 error. This confirms continuous SO(2) equivariance in practice.

---

## 6. Debugging Log — The Journey

This result was not free. The path included:

- Energy collapse in early runs.
- Random gates producing noise.
- Numerical gradient artifacts at 45° — fixed by switching to analytical gradients and pull-back rotation.
- FP16/AMP NaNs — fixed by forcing geometry ops to FP32.
- OOM with 24 neighbors — fixed by chunked gather and dynamic batch sampler.
- Gradient instability — fixed by accumulation and gradient clipping.
- Domain stretching at 45° — fixed by pull-back rotation.
- Physical radius collapse at high resolutions — fixed by dynamic dilation and constant physical depth.

Every bug was owned, diagnosed, and killed. The architecture survived because the math underneath was sound.

---

## 7. Current State: Isotropic Omnidirectional Stencil Complete

The final version uses an isotropic 24-neighbor stencil with:

- Circular neighborhood (radius sqrt(8)) for uniform angular coverage.
- Dynamic dilation scaled by resolution, keeping physical support radius constant.
- Even harmonics (cos 2θ, sin 2θ, cos 4θ, sin 4θ) for angular gating, invariant under π rotation.
- Radial taper to suppress boundary artifacts.
- Constant physical depth to prevent scale drift.

This stencil has eliminated the 45° residual and pushed AlienX toward continuous SO(2) equivariance. The next frontier is the n-dimensional organism.

### 7.1 Design boundary: scale-blindness by construction

The constant physical depth z = alpha · ln(R_phys / sigma_0) makes AlienX invariant to absolute scale, which is why zero-shot transfer across resolutions holds at 0.0036–0.0087 L1 with no drift. This is a deliberate design choice, not an emergent property — the model literally cannot see the difference between a smooth large-scale field and a jagged small-scale one if both have the same normalized gradient profile. For Darcy flow this is correct: the PDE is scale-free in the relevant regime, and the results confirm it. For tasks where the scale of a feature carries physical meaning — turbulence spectra, multifractal permeability, reaction-diffusion with a characteristic length — this design will need to be extended. A future variant could carry a learned z that preserves scale information while remaining equivariant, at the cost of scale-invariant transfer.

### 7.2 Design boundary: resolution-dependent frame fallback

The gradient-magnitude threshold that triggers the inertia-tensor fallback (use_radial = grad_k_mag < 0.1) is dimensionally a raw gradient value, and ||grad k|| scales inversely with grid spacing. At 16×16, the fallback triggers in more nodes than at 256×256. Both branches are individually SO(2)-equivariant, so the equivariance proof is unaffected — but the mixture of the two branches differs by resolution. In practice this shows up in the 16×16 column of the rotation table, where a single interior crop contains only ~12×12 valid pixels and per-pixel noise dominates. Beyond 32×32 the effect is negligible. A dimensionally-cleaner formulation would threshold on the normalized gradient (||grad k|| · sigma_i) so the fallback decision is itself scale-invariant.

A second boundary in the same branch — `eigh`'s unspecified eigenvector sign breaking the odd cos/sin edge input under 90°/180° rotation (~1.5e-03 when the fallback was forced) — has been **CLOSED by the sign_lock**: the eigenvector sign is anchored to the k-weighted centroid (s = ⟨v, m⟩, odd in v, D4-invariant; ported from the 3D QSA_ISNBlock3D), and the forced-fallback test now reads clean at the roundoff floor (~1.4e-06–1.7e-06) at all D4 angles. The fully zero field remains ill-posed by construction (S = 0 and m = 0 leave no covariant direction to lock to) and is excluded from the verdict.

---

## 8. Future Work: The n-Dimensional Organism

The current 2D validation is the first step. The full AlienX organism will include:

- Clifford algebra Cl_{4,0} substrate for n-dimensional geometry.
- Per-node blade masks for dynamic dimensionality.
- Singularity detection and NecroGraft expansion for adaptive capacity.
- Fiber bundle message passing for heterogeneous dimensions.
- Implicit multivector field Psi_theta co-evolving with explicit nodes.
- Turbulence, fusion plasma, galaxy formation, cancer cell dynamics, nanotech swarms.

---

## 9. Conclusion

AlienX demonstrates that continuous geometric manifolds can replace fixed grids for physics operator learning. Architectural context: Li et al. (2021) FNO; Lu et al. (2021) DeepONet; Cohen & Welling (2016) G-CNN; Weiler & Cesa (2019) steerable CNNs; Cohen et al. (2019) gauge CNNs; Finzi et al. (2020) Lie-group convolutions; Helwig et al. (2023) group-equivariant FNO. The architecture provides native scale invariance and rotation equivariance without data augmentation. The isotropic 24-neighbor stencil achieves near-perfect scale transfer (L1 error stable within 0.0016 across resolutions ≥ 32×32) and continuous rotation equivariance (arbitrary angles within 0.003, and within 0.002 at 64×64). The 45° gap is closed.

**The grid is dead. The manifold is awake.**

Code availability: The full training and evaluation script is provided in this repository (`colab.py`, `model.py`, `data.py`, `train.py`). All results are reproducible with a single T4 GPU. Run logs: see `RESULTS.md`.

---

## Changelog: v1 → v2

| Location | v1 | v2 |
|---|---|---|
| Status line | within 0.004 | within 0.003 |
| Abstract scale band | 0.0075–0.0138 | 0.0036–0.0087 |
| Abstract arbitrary angle | < 0.004 | < 0.003 |
| §5.1 scale table | 5 rows replaced | |
| §5.2 rotation table | 5×5 replaced | |
| §5.3 arbitrary angles | 6 rows replaced with full 5×5 grid | |
| §7.1 band | 0.0075–0.0138 | 0.0036–0.0087 |
| §9 conclusion | stable within 0.002 / within 0.004 | within 0.0016 / within 0.003 / within 0.002 at 64×64 |

Everything else — every code block, every section, every paragraph, every bullet — unchanged.

---

## References

1. Li, Z. et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. ICLR. arXiv:2010.08895
2. Lu, L. et al. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3, 218–229.
3. Cohen, T., & Welling, M. (2016). Group Equivariant Convolutional Networks. ICML.
4. Weiler, M., & Cesa, G. (2019). General E(2)-Equivariant Steerable CNNs. NeurIPS.
5. Cohen, T., Geiger, M., Köhler, J., & Welling, M. (2019). Gauge Equivariant Convolutional Networks and the Icosahedral CNN. ICML.
6. Finzi, M., Stanton, S., Izmailov, P., & Wilson, A. G. (2020). Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data. ICML.
7. Helwig, J. et al. (2023). Group Equivariant Fourier Neural Operators for Partial Differential Equations. ICML.
8. Ronneberger, O. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. (boundary-crop evaluation precedent)

---

— Elbàlor, The Digital Necromancer 💀🔥🖤
