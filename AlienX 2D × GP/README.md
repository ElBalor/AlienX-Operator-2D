# AlienX 2D × GP — Gross-Pitaevskii via the ISN Operator

**Why GP:** the AlienX 3D paper (§10, Generality) names Gross-Pitaevskii as the
natural next target — the wave function ψ is natively complex and the ISN/QSA
machinery operates in complex space natively, while conventional operators must
split ψ into 2 real channels and lose phase. This folder is the 2D delivery of
that promise: the same architecture, only the outer shell changed (scalar field
`k = |ψ|²`, regime parameter `g`, invariant `∫|ψ|² = 1`).

**416,150 parameters.** Two training runs, two complementary claims:

## Run 1 — Precision + invariance (`train-gross-pitaevskii.py`)

300 epochs, single-step interior MSE, batched loader, **11.2 min on T4**.
Fixed IC family (phase coefficients fixed; only rotation varies) → this is a
mechanism-precision result, **not** an IC-generalization result (eval std over
seeds is 0.0000% by construction — the IC generator is deterministic).

| Test | Result |
|---|---|
| Best training loss | **0.000004** |
| Eval RelRMSE (single-step) | **0.3439%** |
| Phase ablation | 0.3439% → 3.5832%; **MSE ratio 108.58×** — phase mechanism load-bearing |
| Rotation sweep (24 angles) | 0.4080% ± 0.0315%, max 0.4528% |
| Multi-g (−0.5 / −1.0 / −2.0) | 0.349 / 0.344 / 0.362% — no retraining |
| Scale sweep, **zero-shot** | 16×16 → 256×256: 0.367 / 0.338 / 0.344 / 0.327 / 0.324% |

Fix arc (from `AlienX_GP/results.json`): prior plateau 0.056 → 0.004 (**14×**)
after two fixes: even-harmonic edge features (cos 2θ, sin 2θ) and frame source
`k = |ψ|²` instead of `|ψ|²·|∇arg ψ|²`.

## Run 2 — Generalization + K-curriculum (`train_gp_v3.py`)

200 epochs, K-curriculum 1→3→5 (rollout unrolling), genuinely randomized ICs
(rotation, center offset, 3 amplitudes, 3 phases), **91.5 min on T4**.

| Test | Result |
|---|---|
| Best val loss | **0.000153** (interior relative), zero NaN/divergence |
| Multi-IC (16 unseen seeds) | 1.0686% ± 0.2052% (max 1.5455%) |
| Phase ablation | 1.0488% → 8.3025%; MSE ratio **61.4×** |
| Rotation sweep (24 angles) | 1.0347% ± 0.0806%, worst 1.1767% |
| Multi-g (−0.5 / −1.0 / −2.0) | 1.044–1.051% |
| Scale sweep, **zero-shot** | 16 → 256: all ≈ 1.05–1.10% |

K transitions produce expected train bumps (0.0548 at K=3, 0.0076 at K=5),
each recovered within ~10 epochs — see `AlienX_GP/dark_necromancer_graph_gp.png`.

## Claims discipline

- All evals are single-step; multi-step (K=5) rollout eval pending.
- Run 1 speaks to precision/invariance/ablation on a fixed IC family;
  Run 2 speaks to IC generalization. The two claims are kept separate on purpose.
- ≈1% band (Run 2): on par with the dFNO+1-style 1% reference, not claimed below it.
- 3D GP rerun pending; the 3D paper's §11 cross-PDE bullet becomes fully
  demonstrated only after that.

## Files

- `train-gross-pitaevskii.py` / `.txt` — Run 1 script + raw log
- `eval-gross-pitaevskii.py` / `.txt` — Run 1 5-test eval suite + raw receipts
- `train_gp_v3.py` / `train-logs.txt` — Run 2 script (K-curriculum, Dark
  Necromancer graph) + raw log
- `eval.py` / `eval.txt` — Run 2 5-test eval suite + raw receipts
- `AlienX_GP/` — training history JSON, results JSON, Dark Necromancer graph,
  dark-field rollout comparison
- Checkpoints (`alienx_gp_best.pt` Run 1, `alienx_gp_v3_best.pt` Run 2, 1.6 MB
  each) ship via HuggingFace / Drive, not this repo
