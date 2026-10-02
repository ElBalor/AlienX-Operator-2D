# AlienX 2D × GP — Gross-Pitaevskii via the ISN Operator

The AlienX 2D ISN operator (isotropic 24-neighbor stencil + angular-phase
gating) applied to the 2D Gross-Pitaevskii equation (complex Schrödinger with
nonlinear interaction term), solved as a learned one-step delta predictor.

**416,150 parameters.**

## Training (Colab T4, 91.5 min)

- 192 train / 36 val trajectories; ICs randomized (rotation angle, phase
  coefficients, center offset); resolutions 16 / 32 / 64
- K-curriculum (rollout unrolling): K=1 (epochs 0–40), K=3 (40–100), K=5 (100–200)
- Each K transition causes an expected train bump, fully recovered within ~10 epochs
- **Best val loss: 0.000153** (interior relative), zero NaN / divergence
- Checkpoint `alienx_gp_v3_best.pt` (1.6 MB) ships via HuggingFace / Drive,
  not in this repo

## Full eval (`eval.py`, single-step, interior RelRMSE)

| Test | Result |
|---|---|
| Phase ablation | 1.0488% → 8.3025% when ablated; **MSE ratio 61.4×** — phase mechanism is load-bearing |
| Multi-IC (16 unseen seeds) | 1.0686% ± 0.2052% (max 1.5455%) |
| Multi-g regime (−0.5 / −1.0 / −2.0) | 1.044–1.051%, no retraining |
| Rotation sweep (24 angles, 0–345°) | 1.0347% ± 0.0806%, worst 1.1767% |
| Scale sweep, **zero-shot** | 16×16 → 256×256 all ≈ 1.05–1.10% |

## Claims discipline

- All eval numbers are **single-step**. Multi-step (K=5) rollout eval pending.
- ≈1% RelRMSE band: on par with the dFNO+1-style 1% reference, not claimed below it.
- 3D GP rerun pending; paper claims wait for that verdict.

## Files

- `train_gp_v3.py` — training script (K-curriculum, Dark Necromancer graph)
- `eval.py` — 5-test evaluation suite (ablation / multi-IC / multi-g / rotation / scale)
- `train-logs.txt`, `eval.txt` — raw run receipts
