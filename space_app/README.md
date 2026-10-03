---
title: AlienX 2D Sigil Lab
emoji: 🔮
colorFrom: gray
colorTo: green
sdk: gradio
app_file: app.py
pinned: true
license: agpl-3.0
short_description: Rotate the world — this Darcy-flow operator does not notice
tags:
  - neural-operator
  - equivariance
  - darcy-flow
  - so2-equivariant
  - scientific-ml
---

# AlienX 2D · Sigil Lab

Darcy flow through a **physically rotated** permeability field. Drag the
rotation slider to any angle and the interior L1 error does not move —
rotation is a coordinate artifact, removed before the first layer. Switch
resolution (16² → 128²) with the same weights; **Cast the Compass** sweeps
all 12 angles and plots the flat line that is the entire claim.

- **Model & paper:** [ElBalor/AlienX-2D-Isotropic-Stencil](https://huggingface.co/ElBalor/AlienX-2D-Isotropic-Stencil)
- Isotropic 24-neighbor stencil (radius √8) — no cardinal spikes, no 45° residual
- D4 equivariance test: **bitwise exact** (0.0e+00 deviation) at 90°/180°/270°
- Zero-shot scale transfer: 256² never seen in training, same error band
- One error table per angle flat within **< 0.003** across 0°–199°

*The grid is dead. The manifold is awake.* — from the Grimoire of Elbàlor
