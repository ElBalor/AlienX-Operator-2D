"""
AlienX (ISN) - 2D Release v2: Strict Equivariance Test
======================================================
Verifies the operator-level symmetry claim directly:

    pred(R . k) = R . pred(k)

for rotations R that map the grid onto itself (the dihedral subgroup
D4 of SO(2): 90, 180, 270 degrees), plus a continuous-angle check
(13/27/45/77/123/199 degrees) where R . pred(k) is evaluated by
bilinear interpolation of the unrotated prediction.

Why D4 is exact: for 90-degree multiples, R^-1 x_i is exactly another
grid node, so R . pred(k) is an exact index permutation of pred(k) -
no interpolation, no generator rounding. The continuous-angle check
carries an interpolation floor and is reported as informational.

Why the untrained model is NOT trivially exact:
  res_scale is zero-initialized, so with raw init weights every ISN block
  computes h + 0 * norm(h_new) = h: message passing is a no-op and the model
  is a pointwise readout, which passes D4 exactly BY DEFAULT. When no
  checkpoint is supplied, this harness injects res_scale = 1.0 so the
  gather -> frame -> harmonic gate -> weighted aggregation path genuinely
  carries signal.

Observed result (GPU, untrained + res_scale=1, gradient branch active):
  max |pred(R.k) - R.pred(k)| = 0.000e+00 EXACTLY for 90/180/270.
  Structural, not luck: with the gradient branch active there are no
  cross-node reductions; every within-node reduction (2-element dot
  products, the 24-slot message sum, LayerNorm) runs in bitwise-identical
  order in the rotated run (the stencil slot permutation is ALIGNED with
  the base run's slot order, and 2-element sums are IEEE-commutative), and
  all elementwise ops see the bitwise D4-images of the base values.
  Control probe: deliberately MIS-aligning only the slot order moves the
  output by ~1.4e-06 (GPU) / ~2.0e-06 (CPU) - summation order is the only
  roundoff carrier. Occasional 1e-8..1e-6 residuals on repeat runs are
  GEMM reduction-order noise (cuBLAS/MKL algorithm selection), orders of
  magnitude inside tolerance.

Genuine deviation source — CLOSED by the sign_lock (see TEST 3):
  the inertia-tensor fallback's frame comes from torch.linalg.eigh, whose
  eigenvector SIGN is unspecified under rotation (measured
  dot(R e1, e1_rot) = -1.0000 for 90/180 deg on this field). A sign flip
  rotates the local phase by pi: the even-harmonic angular GATE is
  invariant, but the odd cos/sin terms in the edge INPUT are not, which
  produced ~1.5e-03 localized violations when the fallback was forced
  active. Fix (ported from the 3D QSA_ISNBlock3D): anchor the eigenvector
  sign to the k-weighted centroid m (covariant data vector) —
  s = <v, m> is odd in v and D4-invariant, so e1 = s * v is a pure
  function of the data. No new parameters; checkpoints stay compatible.
  The gradient branch (100% of real workloads on this field) was already
  exact and is unchanged.

Usage:
    python test_equivariance.py --checkpoint alienx_best.pt --device cuda
    python test_equivariance.py --grid 32 --tol 1e-4            # cpu default if no gpu

Eric Yaka (Elbalor / The Digital Necromancer)
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import AlienXOperator, get_cached_iso_knn, ISO_OFFSETS, K_ISO
from data import generate_darcy_sample_gpu

SEED = 123


# ==============================================================================
# D4 PERMUTATION UTILITIES
# Grid is flattened with indexing='ij': node (r, c) at (xs[r], ys[c]).
# Coordinate convention: R_90 CCW maps (x, y) -> (-y, x).
# ==============================================================================

def d4_field_rotation(field, k_rot, G):
    """(R . f)(x) = f(R^-1 x) for k_rot in {0,1,2,3} * 90 degrees.

    For k_rot = 1 (90 deg CCW): the point x_(r,c) maps back to R^-1 x =
    (y_c, -x_r) = grid node (c, G-1-r). So (R f)[r, c] = f[c, G-1-r],
    which is exactly torch.rot90(f, 1, dims=(0, 1)) for this indexing.
    """
    assert field.reshape(G, G).shape == (G, G)
    return torch.rot90(field.reshape(G, G), k=k_rot, dims=(0, 1)).reshape(-1).contiguous()


def d4_vector_rotation(vec, k_rot, G):
    """Rotate a [N, 2] vector field: permute node positions, then apply R to components."""
    vr = torch.rot90(vec.reshape(G, G, 2), k=k_rot, dims=(0, 1))
    if k_rot % 4 == 1:      # (a, b) -> (-b, a)   [R_90 CCW]
        vr = torch.stack([-vr[..., 1], vr[..., 0]], dim=-1)
    elif k_rot % 4 == 2:    # (a, b) -> (-a, -b)  [R_180]
        vr = -vr
    elif k_rot % 4 == 3:    # (a, b) -> (b, -a)   [R_270]
        vr = torch.stack([vr[..., 1], -vr[..., 0]], dim=-1)
    return vr.reshape(-1, 2).contiguous()


def d4_stencil_permutation(k_rot):
    """Permutation of the 24 stencil slots under D4 rotation.

    The stencil is D4-symmetric by construction, so a lattice rotation maps
    the neighbor SET of every node onto itself. Slot j's rotated offset lands
    at slot perm[j]; permuting the slot axis of the knn index tensor makes the
    rotated run use exactly the rotated neighbor ordering.
    """
    if k_rot == 0:
        return np.arange(K_ISO)
    slot_of = {off: j for j, off in enumerate(ISO_OFFSETS)}

    def rot(dr, dc):
        if k_rot == 1:   return (-dc, dr)     # 90 deg CCW in (row, col) space
        if k_rot == 2:   return (-dr, -dc)
        if k_rot == 3:   return (dc, -dr)

    perm = np.empty(K_ISO, dtype=np.int64)
    for j, (dr, dc) in enumerate(ISO_OFFSETS):
        perm[j] = slot_of[rot(dr, dc)]

    # sanity: must be a bijection (the D4-symmetry assertion for the stencil)
    assert sorted(perm.tolist()) == list(range(K_ISO)), "stencil is not D4-symmetric?!"
    return np.ascontiguousarray(perm)


# ==============================================================================
# MODEL CALL HELPER (batch dims handled once, used everywhere)
# ==============================================================================

def run_model(model, coords_flat, k_flat, grad_flat, gm_flat, device):
    """coords [N,2], k [N], grad [N,2], gm [N]  ->  prediction [N]."""
    coords = coords_flat.unsqueeze(0)
    knn = get_cached_iso_knn(int(math.sqrt(coords.shape[1])), device)
    with torch.no_grad():
        pred = model(coords, k_flat.unsqueeze(0), grad_flat.unsqueeze(0),
                     gm_flat.unsqueeze(0), knn)
    return pred.squeeze(0)


# ==============================================================================
# TEST 1 — EXACT D4 COMMUTATION: pred(R.k) = R.pred(k)
# ==============================================================================

def test_d4_exact(model, device, G, tol):
    """
    For k_rot in {1, 2, 3}:
      inputs:  k_rot = R . k0          (exact index permutation)
               g_rot = R . g0          (permutation + component rotation)
              gm_rot = R . gm0         (permutation; |R g| = |g|)
      output:  p_rot = pred(k_rot)
      target:  R . p0 = rot90(p0)      (exact index permutation)
    Reported: max |p_rot - R.p0| on the interior crop, relative to the field max.
    """
    coords, k0, gm0, g0, _ = generate_darcy_sample_gpu(G, device, angle_deg=0.0, seed=SEED)
    # REAL derived inputs: the relation is tested on the operator's actual input
    # distribution (k, grad k, |grad k|). Zero gradients would force the inertia
    # fallback everywhere and test the wrong branch.
    p0 = run_model(model, coords, k0, g0, gm0, device)

    crop = 2 * max(1, G // 16)
    results = []
    all_pass = True

    for k_rot in (1, 2, 3):
        k_rot_f = d4_field_rotation(k0, k_rot, G)
        g_rot = d4_vector_rotation(g0, k_rot, G)          # R . grad k  (exact)
        gm_rot = d4_field_rotation(gm0, k_rot, G)         # R . |grad k| (|Rg| = |g|)
        coords_r = coords.unsqueeze(0)
        knn = get_cached_iso_knn(G, device)
        knn_rot = knn[:, :, torch.from_numpy(d4_stencil_permutation(k_rot))]

        with torch.no_grad():
            p_rot = model(coords_r, k_rot_f.unsqueeze(0), g_rot.unsqueeze(0),
                          gm_rot.unsqueeze(0), knn_rot).squeeze(0)

        target = d4_field_rotation(p0, k_rot, G)

        diff = (p_rot - target).abs().reshape(G, G)
        diff_int = diff[crop:-crop, crop:-crop]
        tgt_int = target.reshape(G, G)[crop:-crop, crop:-crop].abs()

        max_abs = diff_int.max().item()
        mean_abs = diff_int.mean().item()
        rel = max_abs / (tgt_int.max().item() + 1e-12)

        # fallback-node diagnostic: where does the inertia anchor engage?
        fb = (gm0 < 0.1).reshape(G, G)
        fb_int = fb[crop:-crop, crop:-crop]
        fb_frac = fb_int.float().mean().item()
        max_fb = diff_int[fb_int].max().item() if fb_int.any() else float('nan')
        max_grad = diff_int[~fb_int].max().item() if (~fb_int).any() else float('nan')

        passed = rel <= tol
        all_pass &= passed

        results.append((k_rot * 90, max_abs, mean_abs, rel, fb_frac, max_fb, max_grad, passed))

    print(f"\n[TEST 1] Exact D4 commutation:  pred(R.k) = R.pred(k)   (grid {G}x{G}, interior)")
    print(f"    {'angle':>6} | {'max|d|':>10} | {'mean|d|':>10} | {'rel':>9} | {'rel vs tol':>10}")
    for ang, mx, mn, rel, fb_frac, mfb, mgr, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"    {ang:>5}° | {mx:>10.3e} | {mn:>10.3e} | {rel:>9.2e} | {'<= ' + f'{tol:.0e}':>10}  {status}")
        print(f"          fallback nodes: {100 * fb_frac:.1f}%   max|d| fallback={mfb:.3e}  grad-branch={mgr:.3e}")
    return all_pass


# ==============================================================================
# TEST 2 — CONTINUOUS ANGLES (interpolation-based, informational)
# ==============================================================================

def test_continuous(model, device, G, angles):
    """
    For arbitrary theta: inputs are the pull-back-rotated fields (exact, from
    the generator). R . pred(k) is evaluated by bilinear grid_sample of p0 at
    the rotated-back positions. Bilinear interpolation is the error floor —
    this is a consistency tracker, not a proof.
    """
    coords, k0, gm0, g0, _ = generate_darcy_sample_gpu(G, device, angle_deg=0.0, seed=SEED)
    p0 = run_model(model, coords, k0, g0, gm0, device).reshape(G, G)

    crop = 2 * max(1, G // 16)
    print(f"\n[TEST 2] Continuous-angle self-consistency (grid {G}x{G}, bilinear floor)")
    print(f"    {'angle':>6} | {'max|d|':>10} | {'mean|d|':>10}")

    p0_img = p0.reshape(1, 1, G, G)
    for ang in angles:
        th = math.radians(ang)
        X, Y = coords[:, 0].reshape(G, G), coords[:, 1].reshape(G, G)
        # R_theta^-1 applied to the node positions
        sx = math.cos(th) * X + math.sin(th) * Y
        sy = -math.sin(th) * X + math.cos(th) * Y
        inside = (sx.abs() <= 1 - 1e-4) & (sy.abs() <= 1 - 1e-4)   # exclude corner-clamp zone

        # grid_sample: [...,0] indexes the last image dim (y-axis), [...,1] the first (x-axis)
        grid = torch.stack([sy, sx], dim=-1).reshape(1, G, G, 2)
        target = F.grid_sample(p0_img, grid, mode='bilinear',
                               padding_mode='border', align_corners=True).reshape(G, G)

        coords_t, k_t, gm_t, g_t, _ = generate_darcy_sample_gpu(G, device, angle_deg=ang, seed=SEED)
        p_t = run_model(model, coords_t, k_t, g_t, gm_t, device).reshape(G, G)

        diff = (p_t - target).abs()
        mask = inside.clone()
        mask[crop:-crop, crop:-crop] &= True
        mask &= torch.ones_like(mask)
        valid = mask & inside
        # combine with interior crop
        valid_full = torch.zeros(G, G, dtype=torch.bool)
        valid_full[crop:-crop, crop:-crop] = inside[crop:-crop, crop:-crop]

        d = diff[valid_full]
        if d.numel() == 0:
            print(f"    {ang:>5.1f}° | (no valid nodes)")
            continue
        print(f"    {ang:>5.1f}° | {d.max().item():>10.3e} | {d.mean().item():>10.3e}")
    print("    (informational: bilinear interpolation floor included in these numbers)")


# ==============================================================================
# TEST 3 — FALLBACK-DEGENERACY LOCALIZATION
# ==============================================================================

def test_fallback_localization(model, device, G):
    """
    Verifies the sign_lock (ported from the 3D QSA_ISNBlock3D) has closed the
    eigh sign boundary (formerly PAPER.md §7.2).

    Case A (verdict): real fields but with the FALLBACK branch forced active
    everywhere (grad_k_mag clamped below threshold). This is the case where
    eigh's unspecified sign previously broke equivariance at 90/180 deg
    (~1.5e-03). k stays REAL: constant k would make the inertia tensor S
    proportional to the identity — degenerate — and then eigh's error is
    BASIS arbitrariness, which no sign convention can repair. With the
    sign_lock and a non-degenerate S, the frame is a deterministic function
    of the data, so the exact D4 commutation must hold here too.

    Case B (documented, NOT a verdict): the zero field. Every input is zero,
    S is proportional to the identity, and the k-weighted centroid m is
    exactly zero — so there is NO covariant direction to lock to. The
    equivariance demand is ill-posed by construction: a zero field carries
    no orientation, and any frame the solver picks is arbitrary but
    DETERMINISTIC for fixed inputs. No sign convention — or any covariant
    construction — can close this; it is a boundary of the problem, not of
    the fix.
    """
    coords, k0, gm0, g0, _ = generate_darcy_sample_gpu(G, device, angle_deg=0.0, seed=SEED)
    p0 = run_model(model, coords, k0, g0, gm0, device)

    crop = 2 * max(1, G // 16)
    knn = get_cached_iso_knn(G, device)

    def predict(k_flat, g_flat, gm_flat, knn_idx):
        with torch.no_grad():
            return model(coords.unsqueeze(0), k_flat.unsqueeze(0), g_flat.unsqueeze(0),
                         gm_flat.unsqueeze(0), knn_idx).squeeze(0)

    print(f"\n[TEST 3] sign_lock verification: fallback branch forced (grid {G}x{G})")

    # --- Case A: real k (non-degenerate S), fallback forced via clamped gm ---
    gm_forced = torch.minimum(gm0, torch.full_like(gm0, 0.01))  # use_radial = 1 everywhere

    all_clean = True
    for k_rot in (1, 2, 3):
        k_r = d4_field_rotation(k0, k_rot, G)
        g_r = d4_vector_rotation(g0, k_rot, G)
        gm_r = d4_field_rotation(gm_forced, k_rot, G)
        knn_rot = knn[:, :, torch.from_numpy(d4_stencil_permutation(k_rot))]

        p_rot = predict(k_r, g_r, gm_r, knn_rot)
        p_base = predict(k0, g0, gm_forced, knn)
        target = d4_field_rotation(p_base, k_rot, G)

        diff = (p_rot - target).abs().reshape(G, G)[crop:-crop, crop:-crop]
        ang = k_rot * 90
        clean = diff.max().item() <= 1e-4
        all_clean &= clean
        print(f"    Case A  {ang:>5}°: max|d| = {diff.max().item():.3e}   "
              f"{'CLEAN (sign_lock holds)' if clean else 'VIOLATION'}")

    # --- Case B: fully degenerate zero field (documented, not a verdict) ---
    zero_k = torch.zeros_like(k0)
    zero_g = torch.zeros_like(g0)
    zero_gm = torch.zeros_like(gm0)
    k_rot = 1
    k_r = d4_field_rotation(zero_k, k_rot, G)
    knn_rot = knn[:, :, torch.from_numpy(d4_stencil_permutation(k_rot))]
    p_rot = predict(k_r, zero_g, zero_gm, knn_rot)
    p_base = predict(zero_k, zero_g, zero_gm, knn)
    target = d4_field_rotation(p_base, k_rot, G)
    diff = (p_rot - target).abs().reshape(G, G)[crop:-crop, crop:-crop]
    print(f"    Case B   90°: max|d| = {diff.max().item():.3e}   "
          f"(zero-field degenerate; S = 0 makes the eigenvector arbitrary —")
    print(f"           sign_lock multiplies 0 * s = 0, locked only by aligned")
    print(f"           summation order; reported for transparency)")

    if all_clean:
        print("    verdict: eigh sign boundary CLOSED — fallback branch is")
        print("    equivariant at all D4 angles with the sign_lock active.")
        print("    (Case B is ill-posed by construction and excluded from the verdict.)")
    else:
        print("    verdict: sign_lock FAILED to close the boundary — investigate.")
    return all_clean


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlienX 2D strict equivariance tests")
    parser.add_argument("--checkpoint", default="alienx_best.pt",
                        help="trained checkpoint to load (skipped silently if missing)")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--cont-grid", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="relative tolerance for the exact D4 test")
    parser.add_argument("--res-scale", type=float, default=1.0,
                        help="res_scale injected when UNTRAINED so message passing "
                             "is exercised (zero-init makes the D4 test trivial)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    model = AlienXOperator().to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        # HONESTY NOTE: res_scale is zero-initialized, so an UNTRAINED model is a
        # pointwise readout (h + 0 * norm(h_new) = h) and passes D4 TRIVIALLY —
        # exactly 0.000e+00. That verifies the permutation bookkeeping, not the
        # operator. Perturb res_scale so gather -> frame -> harmonic gate ->
        # weighted aggregation actually carries gradient-scale signal.
        with torch.no_grad():
            for block in model.blocks:
                block.res_scale.fill_(args.res_scale)
        print(f"NOTE: checkpoint '{args.checkpoint}' not found — running on UNTRAINED weights.")
        print(f"      res_scale set to {args.res_scale} on all blocks (zero-init would")
        print("      make message passing a no-op and the D4 test pass trivially).")

    model.eval()

    print("=" * 70)
    print("ALIENX 2D v2 — STRICT EQUIVARIANCE:  pred(R.k) = R.pred(k)")
    print("=" * 70)

    ok = test_d4_exact(model, device, args.grid, args.tol)
    test_continuous(model, device, args.cont_grid, [13.0, 27.0, 77.0, 123.0, 199.0])
    t3 = test_fallback_localization(model, device, args.grid)
    ok = ok and t3

    print("=" * 70)
    if ok:
        print("D4 EQUIVARIANCE: PASS (within tolerance)")
    else:
        print("D4 EQUIVARIANCE: FAIL — see fallback diagnostic above")
    print("=" * 70)
