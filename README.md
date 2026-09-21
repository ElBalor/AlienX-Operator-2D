# The AlienX-Operator, with Isomorphic Spaciel Net (ISN) — 2D Release v2: Isotropic 24-Neighbor Stencil

**Isomorphic Spatial Net** : A neural operator that operates on continuous geometric
manifolds instead of fixed grids, with native rotation and scale invariance.

> Strict operator-level verification: **pred(R·k) = R·pred(k), bitwise exact**
> under D4 (90/180/270°) with live message passing — fallback branch closed at
> the roundoff floor by the sign_lock. Rerun it yourself:
> `python test_equivariance.py --device cuda`

**The grid is dead. The manifold is awake.**

Creted by Eric Heylel Danjuma Yaka— Capital Software / Next Gen Tech.


## What's in this release

| File | Purpose |
|---|---|
| `model.py` | `AlienXOperator` + `ISNBlock` + isotropic 24-neighbor stencil (importable module) |
| `data.py` | GPU Darcy flow generator with pull-back rotation and analytical gradients |
| `train.py` | Training + full evaluation suite (modular, imports model/data) |
| `colab.py` | Single-file self-contained version — paste into Colab, identical to the as-run script |
| `test_equivariance.py` | Strict operator-level equivariance test: `pred(R·k) = R·pred(k)` |
| `RESULTS.md` | Full run logs for two independent v2 runs + v1→v2 comparison + strict equivariance results |
| `PAPER.md` | The complete v2 paper |

## The architecture in one paragraph

Each node lives on a spatial graph with an **isotropic 24-neighbor circular stencil**
(radius √8 — uniform angular coverage, no cardinal spikes). A **local SO(2) frame**
(e1 from ∇k, inertia-tensor fallback where ∇k is weak) makes displacements
rotation-equivariant. Edge features encode the local angle as a **complex harmonic**
embedding with even harmonics (cos 2θ, sin 2θ, cos 4θ, sin 4θ) — message weights are
invariant under θ → θ + π. A **constant physical scale depth** makes the operator
scale-blind by construction, which is exactly what produces zero-shot transfer across
resolutions. No data augmentation. No learned geometry.

## Results (v2, Interior L1)

| Test | Result |
|---|---|
| Scale invariance (16→256, 256 zero-shot) | 0.0036–0.0087, no drift |
| Rotation equivariance (0/45/90/180/270°) | essentially identical — 45° residual eliminated |
| Arbitrary angles (13/27/77/123/199°) | < 0.003 deviation at ≥ 32×32; < 0.002 at 64×64 |

Full logs in `RESULTS.md`.

## Strict equivariance: pred(R·k) = R·pred(k)

The evaluation suite measures accuracy of the rotated problem. The strict
statement — rotating the input rotates the output identically — is tested
directly in `test_equivariance.py`:

```bash
python test_equivariance.py --checkpoint alienx_best.pt --device cuda
```

Three tests:

1. **Exact D4 commutation** (90/180/270°): `R·k` and `R·pred(k)` are exact
   index permutations of the flat fields — no interpolation, no generator
   rounding. Result on the v2 architecture with message passing live:
   **max |pred(R·k) − R·pred(k)| = 0.000e+00 — bitwise exact.** With the
   gradient branch active there are no cross-node reductions, and every
   within-node reduction runs in bitwise-identical order in the rotated
   run (aligned slot permutation; 2-element sums are IEEE-commutative).
   Control probe: mis-aligning only the slot order moves the output by
   ~1.4e-06 — summation order is the sole roundoff carrier, and it is
   aligned. (Earlier draft's "1–2e-08" was an artifact of zero-init
   res_scale — messages were a no-op; retracted.)
2. **Continuous angles** (13/27/77/123/199°): evaluated against bilinearly
   interpolated `R·pred(k)` — max ≈ 1e-01, mean ≈ 5e-03–8e-03, uniformly
   flat across angles (interpolation + discretization floor, not operator
   error). Informational regression tracker.
3. **`sign_lock` verification**: the eigh sign boundary is CLOSED.
   `eigh`'s eigenvector sign is unspecified under rotation
   (dot(R·e1, e1_rot) = −1.0000 at 90°/180°), which previously broke the
   odd cos/sin edge input at ~1.5e-03 when the fallback was forced. Fix
   (ported from the 3D QSA_ISNBlock3D): anchor the eigenvector sign to the
   k-weighted centroid — s = ⟨v, m⟩ is odd in v (cancels the arbitrary
   flip) and D4-invariant. With the fallback branch forced active, all
   D4 angles read CLEAN at the roundoff floor (~1.4e-06 – 1.7e-06).
   The gradient branch — 100% of real workloads on this field — is exact.

```text
[TEST 1] Exact D4 commutation:  pred(R.k) = R.pred(k)   (grid 32x32, interior)
     angle |     max|d| |    mean|d| |       rel
        90° |  0.000e+00 |  0.000e+00 |  0.00e+00  PASS
       180° |  0.000e+00 |  0.000e+00 |  0.00e+00  PASS
       270° |  0.000e+00 |  0.000e+00 |  0.00e+00  PASS

[TEST 3] sign_lock verification: fallback branch forced (grid 32x32)
    Case A     90°: max|d| = 1.431e-06   CLEAN (sign_lock holds)
    Case A    180°: max|d| = 1.669e-06   CLEAN (sign_lock holds)
    Case A    270°: max|d| = 1.669e-06   CLEAN (sign_lock holds)
    Case B   90°: max|d| = 1.315e-03   (zero-field degenerate; S = 0 makes
           the eigenvector arbitrary — ill-posed by construction, excluded)
    verdict: eigh sign boundary CLOSED — fallback branch is
    equivariant at all D4 angles with the sign_lock active.
```

Note: with no checkpoint present, the harness injects `res_scale = 1.0`
into all blocks — zero-init would make message passing a no-op and the
D4 test would pass trivially (exactly 0). The injection makes the test
exercise the real gather → frame → harmonic gate → aggregation path.

## Reproduce

```bash
# Single T4 GPU, ~15 minutes for 500 epochs
python colab.py

# or modular:
python train.py   # trains, evaluates, saves alienx_rotation_continuous.png
```

Config: AdamW lr 2e-3, weight decay 1e-4, cosine annealing over 500 epochs,
resolutions {16, 32, 64, 128} with 16 samples each, random rotations in [0, 360),
interior MSE with dilation-scaled boundary crop. Best checkpoint: `alienx_best.pt`.

## The debugging log (no failures hidden)

- Energy collapse in early runs
- Random gates producing noise
- Numerical gradient artifacts at 45° → analytical gradients + pull-back rotation
- FP16/AMP NaNs → geometry ops forced to FP32
- OOM with 24 neighbors → chunked gather + dynamic batch sampler
- Gradient instability → accumulation + gradient clipping
- Domain stretching at 45° → pull-back rotation
- Physical radius collapse at high resolutions → dynamic dilation + constant physical depth

Every bug was owned, diagnosed, and killed.

## Known design boundaries (documented, not hidden)

1. **Scale-blindness by construction** — correct for Darcy flow (scale-free PDE),
   needs extension for scale-carrying physics (turbulence spectra, multifractal
   permeability). See PAPER.md §7.1.
2. **Resolution-dependent frame fallback** — the `grad_k_mag < 0.1` threshold is a
   raw gradient value; both branches are equivariant, but the branch mixture differs
   by resolution. Cleaner formulation: threshold on ||∇k||·σ. The former eigh
   sign-flip boundary *inside* the fallback branch is CLOSED by the sign_lock
   (TEST 3: clean at the roundoff floor at all D4 angles). See PAPER.md §7.2.

## Next: the n-Dimensional Organism

Clifford algebra Cl(4,0) substrate, per-node blade masks, NecroGraft expansion,
fiber-bundle message passing. The 3D frontier (SO(3)) lives in `../AlienX-S03-Invariance/`.

## License

From The Grimoire of Elbàlor The Digital Necromancer.

AGPL-3.0 — see [LICENSE](LICENSE).
