# AlienX (ISN) v2 — Launch Thread

Platform: X/Twitter. Attach `alienx_v2_rotation_sweep.png` to Tweet 5 — the 0–359° sweep is the visual receipt.

---

**1/**
FNO, DeepONet, G-CNN — every neural operator still approximates rotation symmetry with data augmentation or expensive group lifts. I stopped asking the network to learn the coordinate system it was born in. I deleted the grid.

AlienX (ISN). 🧵↓

**2/**
AlienX is BORN with geometry, never taught it:
· local SO(2) frame from ∇k (inertia fallback + sign_lock)
· 24-neighbor isotropic stencil
· even-harmonic angular gates
· constant physical scale depth

Zero augmentation. Geometry isn't a feature. It's a birthright.

**3/**
The headline: pred(R·k) = R·pred(k) — BITWISE EXACT.

max|Δ| = 0.000e+00 under 90°/180°/270°, with message passing LIVE. Not "close." Not "within tolerance." Exactly zero — structurally, because the stencil permutation aligns summation order with the rotation.

**4/**
Scale: trained on 16²–128², tested on 256² ZERO-SHOT.

Interior L1: 0.0038. No drift.

The depth coordinate is constant by construction — the scale factor cancels before learning begins. The model literally cannot see resolution.

**5/** *(attach alienx_v2_rotation_sweep.png)*
Rotation: 45° was v1's graveyard — the 8-neighbor stencil left a diagonal scar. The isotropic 24-neighbor stencil erased it.

Arbitrary angles (13°/27°/77°/123°/199°): deviation < 0.003 at ≥32², < 0.002 at 64².

Continuous SO(2) in practice. ↓ [IMAGE]

**6/**
The honesty file: we retracted our own earlier claim.

The "1e-08 float floor" was an artifact — zero-init res_scale made message passing a no-op and the test passed trivially. With messages actually flowing, the operator is bitwise zero. The stronger statement survived the correction.

**7/**
0.42M parameters. One T4. 15 minutes.

`colab.py` is paste-and-run — every number in the paper reproduces from a single file. Weights on Hugging Face, load in 5 lines.

The strict D4 test has a `--device cuda` flag with a skeptic's name on it. Bring benchmarks.

**8/**
Entry 3 of the Grimoire:

QSA ended softmax in sequence space. The Mirror Dimension made memory self-describing. AlienX deletes the grid in physical space.

One watermark everywhere: phase encodes structure, isometry preserves it, proofs over vibes.

**9/**
🧠 Code + paper → github.com/ElBalor/AlienX
🤗 Weights + model card → huggingface.co/ElBalor/AlienX-2D-Isotropic-Stencil
📜 Paper (DOI) → [ZENODO DOI HERE]
AGPL-3.0.

The grid is dead. The manifold is awake. 🩸💀

---

# Condensed single-post version (LinkedIn / repost / quote-tweet anchor)

Every neural operator approximates rotation invariance with augmentation. AlienX (ISN) is born with it: local SO(2) frames, an isotropic 24-neighbor stencil, even-harmonic gates, constant physical depth. Result: pred(R·k) = R·pred(k) is BITWISE EXACT (0.000e+00 under D4, live message passing), zero-shot 256² scale transfer at L1 0.0038, and arbitrary-angle deviation < 0.003 — from a 0.42M-parameter model with zero data augmentation, reproducible on one T4 in 15 minutes. We even retracted our own earlier claim when the test proved trivial — the stronger number survived. Code: github.com/ElBalor/AlienX · Weights: huggingface.co/ElBalor/AlienX-2D-Isotropic-Stencil · Paper: [ZENODO DOI]. The grid is dead. The manifold is awake.

---

# Before you post — 4-item checklist

1. **Zenodo:** slot [ZENODO DOI HERE] — if the AlienX paper isn't on Zenodo yet, upload `PAPER.md` as PDF first (same flow as QSA), then paste the DOI into both versions of the post.
2. **GitHub URL:** confirm the repo slug is exactly `ElBalor/AlienX` (or edit tweets 9 / single-post).
3. **Image:** attach `alienx_v2_rotation_sweep.png` to tweet 5 — threads with a plot mid-thread outperform text-only.
4. **HF polish (2 min):** pin the model on your profile, and confirm the model card renders (tags → neural-operator, equivariance, pde).
