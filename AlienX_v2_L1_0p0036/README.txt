AlienX v2 — S-Rank Run
======================

Training loss (final):  0.000034
64×64 interior L1:      0.003631 (best)  /  0.003649 (final)
Zero-shot 256×256 L1:   0.003759
Arbitrary angle max Δ:  < 0.002

Architecture: 415,509 params, 4 ISN blocks, 24-neighbor isotropic stencil,
              ∇k-frame + inertia fallback, constant physical depth.

Files:
  alienx_v2_best.pt              — best checkpoint (SHIP THIS to HF)
  alienx_v2_final.pt             — final epoch (identical to best ± float noise)
  alienx_v2_rotation_sweep.png   — 0-359° continuous rotation plot

Paper status: AlienX (ISN) — continuous rotation- and scale-invariant
              operator on geometric manifolds.
