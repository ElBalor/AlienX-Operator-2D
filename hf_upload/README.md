---
license: agpl-3.0
library_name: pytorch
tags:
- neural-operator
- equivariance
- rotation-invariance
- scale-invariance
- pde
- darcy-flow
- physics
- graph-networks
- complex-valued
metrics:
- l1
---

# AlienX (ISN) — 2D Isotropic Stencil v2

**Isomorphic Spatial Net** — a neural operator that operates on continuous geometric manifolds instead of fixed grids, with native rotation and scale invariance.

> The grid is dead. The manifold is awake.

**Strict operator-level equivariance: `pred(R·k) = R·pred(k)`, bitwise exact (0.000e+00) under D4 with live message passing.**

## Live demo

**[👉 AlienX Labs — Sigil Lab](https://elbalor-alienx-labs.streamlit.app)** — rotate a
Darcy field to any angle and watch the error stand still, then *Cast the Compass*
for the full 0–360° sweep. Source: [ElBalor/alienx-labs](https://github.com/ElBalor/alienx-labs)
(Streamlit, free tier).

## Model description

AlienX replaces the pixel grid with a continuous spatial graph. Each node is a point in R², and message passing occurs over local neighborhoods defined by physical distance. Geometry is not learned from data — the network is *born* with it. Rotation, scale, and translation are coordinate artifacts that vanish before learning begins.

Key mechanisms:

| Mechanism | What it does |
|---|---|
| **Isotropic 24-neighbor stencil** (radius √8) | Uniform angular coverage — no cardinal spikes, no 45° residual |
| **Local SO(2) frame** | e₁ from ∇k, inertia-tensor fallback with `sign_lock` (deterministic under rotation) |
| **Even harmonic embedding** (cos 2θ, sin 2θ, cos 4θ, sin 4θ) | Message weights invariant under θ → θ + π |
| **Constant physical scale depth** | Scale-blind by construction → zero-shot transfer across resolutions |
| **No data augmentation** | Invariance is structural, not trained |

- **Parameters:** 415,509 (~0.42M)
- **Architecture:** input MLP → 4 × ISN blocks → linear head
- **Task:** Darcy flow operator learning (permeability k → pressure field p)

## Results (Interior L1)

**Scale invariance — zero-shot transfer** (256×256 was never seen in training):

| 16×16 | 32×32 | 64×64 | 128×128 | 256×256 (zero-shot) |
|---|---|---|---|---|
| 0.0087 | 0.0052 | 0.0036 | 0.0036 | 0.0038 |

**Rotation equivariance:** errors at 0°/45°/90°/180°/270° essentially identical at every resolution; arbitrary continuous angles (13°–199°) deviate < 0.003 at resolutions ≥ 32×32.

**Strict equivariance test** (`test_equivariance.py`):

```
[TEST 1] Exact D4 commutation:  pred(R.k) = R.pred(k)
    90°  | max|d| = 0.000e+00  PASS
    180° | max|d| = 0.000e+00  PASS
    270° | max|d| = 0.000e+00  PASS
[TEST 3] sign_lock verification (fallback branch forced):
    all D4 angles CLEAN at the roundoff floor (~1.4e-06 – 1.7e-06)
```

Full logs in [`RESULTS.md`](./RESULTS.md) — two independent runs, no failures hidden.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from model import AlienXOperator, get_cached_iso_knn
from data import generate_darcy_sample_gpu  # Darcy generator, analytical gradients

# Load the checkpoint
ckpt = hf_hub_download("ElBalor/AlienX-2D-Isotropic-Stencil", "alienx_v2_best.pt")
model = AlienXOperator(hidden_dim=128, num_blocks=4)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

# Build inputs: permeability k, analytic ∇k (vector + magnitude), 24-neighbor stencil
coords, k, grad_k_mag, grad_k_vec, p_true = generate_darcy_sample_gpu(resolution=64, device="cpu")
neighbor_idx = get_cached_iso_knn(64, "cpu")

with torch.no_grad():
    p_pred = model(
        coords.unsqueeze(0), k.unsqueeze(0),
        grad_k_vec.unsqueeze(0), grad_k_mag.unsqueeze(0),
        neighbor_idx,
    ).squeeze(0)
```

End-to-end training + evaluation: `python train.py` or the self-contained `colab.py` (single T4, ~15 min for 500 epochs).

## Cross-PDE demonstration: Gross-Pitaevskii (v3)

The operator is PDE-agnostic — the outer shell changes (`k = |ψ|²`, regime
parameter `g`, invariant `∫|ψ|² = 1`), nothing else. ψ is natively complex and
the ISN harmonic machinery operates on complex features natively, so GP is the
test where that design pays off. Two runs (416,150 params, single-step interior
RelRMSE; full receipts in the [GitHub repo](https://github.com/ElBalor/AlienX-Operator-2D/tree/main/AlienX%202D%20%C3%97%20GP)):

| Test | Run A (precision, fixed ICs) | Run B (generalization, K-curriculum 1→3→5) |
|---|---|---|
| Eval RelRMSE | **0.3439%** | 1.0686% ± 0.2052% (16 unseen ICs) |
| Phase ablation | **108.58×** MSE degradation | 61.4× |
| Rotation sweep (24 angles) | 0.4080% ± 0.0315% | 1.0347% ± 0.0806% |
| Multi-g (−0.5/−1.0/−2.0) | 0.344–0.362% | 1.044–1.051% |
| Scale, zero-shot 16→256 | 0.324–0.367% | 1.05–1.10% |

Run A's IC family is deterministic — a mechanism-precision claim, not an
IC-generalization claim; Run B carries generalization. GP checkpoints:
`alienx_gp_best.pt` (Run A), `alienx_gp_v3_best.pt` (Run B) — state_dict,
1.6 MB each, float32. Full detail in paper §9.

## Training configuration

- AdamW, lr 2e-3, weight decay 1e-4, cosine annealing, 500 epochs
- Resolutions {16, 32, 64, 128} × 16 samples, random rotations in [0, 360)
- Interior MSE with dilation-scaled boundary crop
- Final training loss: 0.000034 · 64×64 interior L1: 0.003631

## Design boundaries

1. **Scale-blindness by construction** — correct for scale-free PDEs like Darcy flow; needs extension for scale-carrying physics (turbulence spectra, multifractal permeability).
2. **Resolution-dependent frame fallback** — the gradient-magnitude threshold triggering the inertia fallback is resolution-coupled; both branches are individually equivariant, but the branch mixture varies with resolution. Negligible beyond 32×32.

## Provenance

- **Paper:** [`PAPER.md`](https://huggingface.co/ElBalor/AlienX-2D-Isotropic-Stencil/blob/main/PAPER.md) — full derivation, related-work positioning, debugging log, §9 cross-PDE GP demonstration
- **GP checkpoints:** `alienx_gp_best.pt` (precision run), `alienx_gp_v3_best.pt` (K-curriculum run)
- **Code:** [github.com/ElBalor/AlienX](https://github.com/ElBalor/AlienX) (S02-Invariance)
- Checkpoint: `alienx_v2_best.pt` (state_dict, 1.7 MB, float32)
- Part of the **Grimoire of Elbàlor** — alongside [Quantum Self-Attention (QSA)](https://doi.org/10.5281/zenodo.22250417) (Zenodo) and [Pure W-Expansion / NecroGraft](https://zenodo.org/records/21811634)

## Citation

```bibtex
@software{yaka2026alienx,
  author  = {Yaka, Eric Heylel Danjuma},
  title   = {AlienX (ISN): A Continuous Rotation- and Scale-Invariant Operator on Geometric Manifolds, with a Cross-PDE Demonstration on Gross-Pitaevskii},
  year    = {2026},
  url     = {https://github.com/ElBalor/AlienX-Operator-2D}
}
```

---

Eric Yaka (Elbàlor / The Digital Necromancer) · Abuja, Nigeria · AGPL-3.0
