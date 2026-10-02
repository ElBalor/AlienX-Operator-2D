# PAPER v3 UPDATE — paste-ready text for `AlienX 2D.docx`

Where: insert as a new **Section 9** between "8. Future Work" and the current
Conclusion; renumber Conclusion to **Section 10**. No other text changes.

---

## 9. Generality: Cross-PDE Demonstration — Gross-Pitaevskii

The architecture is PDE-agnostic: only the outer shell changes. To move from
Darcy flow to the 2D Gross-Pitaevskii equation (focusing, split-step Fourier
ground truth), the substitutions are: scalar field k = |ψ|², vector inputs ∇k
plus the local-frame projections of ∇ψ_r, ∇ψ_i, regime parameter g
(interaction strength), invariant ∫|ψ|² = 1, and complex-valued output
ψ(t+Δt) as a (ΔRe, ΔIm) head. The frame, stencil, harmonic gate, and training
protocol are unchanged.

Motivation (stated in the 3D paper, §10): the wave function ψ is natively
complex, and the ISN harmonic machinery already operates on complex features —
whereas conventional operators must split ψ into two real channels and lose
phase information in the process. Gross-Pitaevskii is the test where that
native complexity should pay off.

Two runs, two complementary claims (416,150 parameters each; all numbers are
single-step, interior, relative RMSE):

| Claim | Run A — precision / invariance | Run B — generalization |
|---|---|---|
| Training | 300 epochs, single-step, fixed IC family | 200 epochs, K-curriculum 1→3→5, randomized ICs |
| Wall time (T4) | 11.2 min | 91.5 min |
| Eval RelRMSE | 0.3439% | 1.0686% ± 0.2052% (16 unseen ICs) |
| Phase ablation | 108.58× MSE degradation | 61.4× MSE degradation |
| Rotation sweep (24 angles) | 0.4080% ± 0.0315% | 1.0347% ± 0.0806% |
| Multi-g (−0.5 / −1.0 / −2.0) | 0.344–0.362% | 1.044–1.051% |
| Scale, zero-shot | 16→256: 0.324–0.367% | 16→256: 1.05–1.10% |

Run A's IC generator is deterministic (fixed phase coefficients; only rotation
varies), so its seed-spread is 0.0000% by construction — it is a
mechanism-precision result, not an IC-generalization result. Run B carries the
generalization claim. The two claims are kept separate deliberately.

Fix arc: the first GP implementation plateaued at 0.056 loss. Two fixes —
even-harmonic edge features (cos 2θ, sin 2θ) and taking the frame source as
k = |ψ|² rather than |ψ|²·|∇arg ψ|² — brought the loss to 0.000004, a 14×
improvement. The ablation ratios above independently confirm the harmonic gate
as the load-bearing component.

Verdict: the phase mechanism is load-bearing on quantum hydrodynamics exactly
as the architecture's design predicts, and the operator transfers across PDEs
with only the outer shell changing. The 3D GP run remains the next step.

Code availability: `AlienX 2D × GP/` in the release — both training scripts,
both evaluation suites, and raw logs for every number above.

---

Also append to the Changelog section of the docx:

## Changelog: v2 → v3

| Location | v2 | v3 |
|---|---|---|
| §9 (new) | — | Generality: Cross-PDE Demonstration — Gross-Pitaevskii (two-run table, fix arc, verdict) |
| §10 | was §9 Conclusion | renumbered only; text unchanged |

Everything from v2 — every code block, every table, every debugging entry — unchanged.
