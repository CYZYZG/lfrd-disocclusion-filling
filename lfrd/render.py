"""Disocclusion filling and postprocessing -- paper III-E.

The paper never inpaints the virtual view directly.  Stage 5 predicts the *occlusion
layer* (the reference region that was removed because it occluded the disocclusion);
this module warps that predicted layer into the virtual view and copies it **only** onto
disocclusion pixels, then cleans up what is left (paper III-E):

1. warp the predicted occlusion layer to the virtual camera (``warp_view``);
2. copy pixels only where the destination is a disocclusion and the sample is valid --
   the paper: "the corresponding region in the virtual image belongs to the
   disocclusion", so valid pixels are never overwritten;
3. postprocessing: the remaining small holes (cracks left over, small holes from depth
   errors) are filled with the same inpainting but with B(p) == 1 ("the restriction of
   the background term is redundant");
4. OOFA is left untouched by default (paper ambiguity #12 in 复现方案.md), ``--fill_oofa``
   fills it as well.

Only predicted samples can win inside a disocclusion: the disocclusion is exactly the set
of virtual pixels that no original reference pixel splats onto, so every sample that lands
there comes from the region that stage 5 re-synthesised.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import inpaint as _inpaint
from . import warp as _warp


def as_bool(mask):
    m = np.asarray(mask)
    return m if m.dtype == bool else (m > 0)


# --------------------------------------------------------------------------- #
# III-E steps 1-2: fill the disocclusion from the predicted occlusion layer
# --------------------------------------------------------------------------- #
def fill_disocclusion(warped_color, warped_depth, hole_disocc, occ_color, occ_depth,
                      cams, src, dst, warped_hole=None):
    """Warp the predicted occlusion layer and copy it onto the disocclusions only.

    Parameters
    ----------
    warped_color : (H,W,3) uint8  plain virtual view (stage 2 output)
    warped_depth : (H,W) uint8/float inverse depth of the plain virtual view
    hole_disocc : (H,W) bool  disocclusion mask of the plain virtual view
    occ_color : (H,W,3) uint8  reference colour with the removed region filled (stage 5)
    occ_depth : (H,W) uint8    predicted inverse depth of the reference view (stage 5)
    cams, src, dst : calibration dict and camera pair
    warped_hole : optional (H,W) bool, hole mask of the plain warp; if given, a pixel is
        only written when it is a hole of the plain warp as well

    Returns
    -------
    ``(composite_color, stats)``.  ``stats`` holds the counts required by III-E plus the
    warped occlusion layer for the visual panel:
    ``filled, no_sample, disocc_px, overwritten`` (must stay 0), ``take`` (the mask that
    was copied), ``occ_color`` (warped), ``occ_depth``, ``occ_hole``, ``composite_depth``.
    """
    warped_color = np.asarray(warped_color, np.uint8)
    hole_disocc = as_bool(hole_disocc)
    res = _warp.warp_view(cams, src, dst, np.asarray(occ_color, np.uint8),
                          np.asarray(occ_depth, np.uint8))
    occ = res["warped_color"]
    occ_valid = ~res["hole"]
    take = hole_disocc & occ_valid
    if warped_hole is not None:
        take &= as_bool(warped_hole)

    out = np.array(warped_color, copy=True)
    out[take] = occ[take]
    changed = (out != warped_color).any(axis=2) if out.ndim == 3 else (out != warped_color)
    overwritten = int((changed & ~hole_disocc).sum())

    wdepth = np.asarray(warped_depth, np.float32)
    cdepth = np.array(wdepth, copy=True)
    occ_d = res["warped_depth"]
    if cdepth.ndim == 3:
        cdepth = cdepth[..., 0]
    sel_d = take & (occ_d >= 0)
    cdepth[sel_d] = occ_d[sel_d]

    stats = dict(
        filled=int(take.sum()),
        no_sample=int((hole_disocc & ~occ_valid).sum()),
        disocc_px=int(hole_disocc.sum()),
        overwritten=overwritten,
        take=take,
        occ_color=occ,
        occ_depth=res["warped_depth"],
        occ_hole=res["hole"],
        composite_depth=cdepth,
    )
    return out, stats


# --------------------------------------------------------------------------- #
# III-E step 3: postprocessing of the remaining holes
# --------------------------------------------------------------------------- #
def postprocess(color, hole_mask, depth, max_area=20000, use_bg_term=False,
                oofa=None, fill_oofa=False, patch_size=9, search_w=160, search_h=120,
                depth_tol=0.2, alpha=255, max_iters=200000, record_source=False):
    """Fill the holes left after III-E step 2 (B(p) == 1), leaving OOFA alone by default.

    ``max_area`` : connected components larger than this are reported in
    ``stats['large_holes']`` / ``large_hole_px`` (they are still filled -- the paper's
    postprocessing has no area limit -- but a big value is a symptom worth reporting).
    ``oofa`` : (H,W) bool mask of the out-of-field area; never written unless ``fill_oofa``.
    """
    color = np.asarray(color, np.uint8)
    hole = as_bool(hole_mask)
    depth = np.asarray(depth, np.uint8)
    todo = hole.copy()
    if oofa is not None and not fill_oofa:
        todo &= ~as_bool(oofa)
    n_todo = int(todo.sum())
    stats = dict(remaining_px=n_todo, oofa_px=int(as_bool(oofa).sum()) if oofa is not None
                 else 0, large_holes=0, large_hole_px=0, filled_px=0, iters=0, seconds=0.0,
                 skipped_oofa_px=int((hole & as_bool(oofa)).sum()) if oofa is not None else 0)
    if n_todo == 0:
        stats["color"] = np.array(color, copy=True)
        stats["depth"] = np.array(depth, copy=True)
        return np.array(color, copy=True), np.array(depth, copy=True), stats

    # report big leftovers (still filled, but a large one is suspicious)
    n, lab, st, _ = cv2.connectedComponentsWithStats(todo.astype(np.uint8), connectivity=8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        big = areas[areas > max_area]
        stats["large_holes"] = int(big.size)
        stats["large_hole_px"] = int(big.sum()) if big.size else 0

    res = _inpaint.inpaint_fill_only(color, todo, depth, patch_size=patch_size,
                                     search_w=search_w, search_h=search_h,
                                     depth_tol=depth_tol, alpha=alpha, max_iters=max_iters,
                                     return_meta=True, record_source=record_source)
    out_c = res["filled"]
    out_d = res["filled_depth"]
    stats.update(filled_px=int(res["n_filled"]), iters=int(res["n_iters"]),
                 seconds=float(res["seconds"]), unfilled=int(res["n_unfilled"]),
                 match_cost_mean=float(res["match_cost_mean"]),
                 branch_hist=res["branch_hist"],
                 candidates_rejected_no_depth=int(res["candidates_rejected_no_depth"]),
                 candidates_rejected_none=int(res["candidates_rejected_none"]))
    if record_source:
        stats["src_map"] = res.get("src_map")
    return out_c, out_d, stats


# --------------------------------------------------------------------------- #
# photometric seam match of the filled region (NOT paper III-E; an add-on)
# --------------------------------------------------------------------------- #
def _seam_bands(hole, valid, ring):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    inner = hole & (cv2.erode(hole.astype(np.uint8), k) == 0)
    outer = valid & (cv2.dilate(hole.astype(np.uint8), k) > 0)
    return inner, outer


def _diffuse(field, hole, known, iters, lam=0.9):
    """Laplace diffusion of a sparse field into `hole` (pure numpy/cv2, no scipy)."""
    f = field.astype(np.float32)
    m = known.astype(np.float32)
    for _ in range(int(iters)):
        avg = cv2.blur(f, (3, 3))
        wavg = cv2.blur(m, (3, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            cand = avg / np.maximum(wavg, 1e-6)
        upd = hole & (wavg > 1e-6)
        if not upd.any():
            break
        f[upd] = (1 - lam) * f[upd] + lam * cand[upd]
        m[upd] = 1.0
    return f


def photometric_seam_match(color, hole, valid, ring=3, clip=25.0, contrast=True,
                           spatial=True, iters=300, strength=1.0):
    """Re-level the filled region to join the surrounding valid content at the seam.

    Two stages, both estimated at the seam only (no ground truth):

    1. **global** -- match the per-channel mean and contrast of the whole filled region to the
       valid content within ``ring`` px of its boundary.  This is what the pipeline shipped first
       and is worth +1.83 dB on BA54.
    2. **spatial** -- re-estimate the (now small) seam offset on the corrected image and diffuse
       it inwards over the hole, removing the LOW-FREQUENCY drift the single constant cannot
       reach.  The residual really is structured: on f000 the per-64px-block offset spans +1 to
       -17 while the global step only removes 6.9.  Worth another **+1.17 dB** (total +3.00 of
       the raw fill), positive on all ten frames.

    Why the pipeline needs this at all: the occlusion layer is synthesised by copying reference
    patches across a camera pair with an SSD match that has no term keeping the copied *level*
    consistent with the virtual view's own background, so the fill drifts.  A smooth region has
    nothing but its level to get wrong, which is why the sibling reproduction -- which fills in
    the virtual view -- was 3.29 dB ahead exactly there.

    Returns ``(color, stats)``.
    """
    out = np.array(color, copy=True)
    hole = as_bool(hole)
    valid = as_bool(valid) & ~hole
    if not hole.any():
        return out, dict(applied=False, reason="no hole")
    inner, outer = _seam_bands(hole, valid, ring)
    if inner.sum() < 20 or outer.sum() < 20:
        return out, dict(applied=False, reason="seam band too small",
                         inner=int(inner.sum()), outer=int(outer.sum()))
    f = out.astype(np.float32)
    gains, shifts = [], []
    for c in range(3):
        hh = f[..., c][inner]
        vv = f[..., c][outer]
        gain = 1.0
        if contrast and hh.std() > 1e-3 and vv.std() > 1e-3:
            gain = float(np.clip(vv.std() / hh.std(), 0.8, 1.25))
        shift = float(np.clip(vv.mean() - gain * hh.mean(), -clip, clip))
        if abs(gain - 1.0) > 1e-6 or abs(shift) > 1e-6:
            f[..., c][hole] = gain * f[..., c][hole] + shift
        gains.append(gain)
        shifts.append(shift)
    f = np.clip(f, 0, 255)
    spatial_stats = None
    if spatial and strength > 0:
        ks = 2 * ring + 1
        cnt = cv2.blur(outer.astype(np.float32), (ks, ks))
        tot = 0.0
        for c in range(3):
            base = f[..., c]
            mean_out = cv2.blur(np.where(outer, base, 0.0).astype(np.float32), (ks, ks))
            with np.errstate(invalid="ignore", divide="ignore"):
                nb = mean_out / np.maximum(cnt, 1e-6)
            sel = inner & (cnt > 1e-6)
            if sel.sum() < 20:
                continue
            field = np.zeros(hole.shape, np.float32)
            known = np.zeros(hole.shape, bool)
            field[sel] = (base - nb)[sel]
            known[sel] = True
            fld = np.clip(_diffuse(field, hole, known, iters), -clip, clip) * float(strength)
            f[..., c][hole] = np.clip(base[hole] - fld[hole], 0, 255)
            tot += float(np.abs(fld[hole]).mean())
        spatial_stats = dict(iters=int(iters), strength=float(strength),
                             mean_abs_field=tot / 3.0)
    out = np.clip(np.rint(f), 0, 255).astype(np.uint8)
    stats = dict(applied=True, ring=int(ring), gains=gains, shifts=shifts,
                 inner_px=int(inner.sum()), outer_px=int(outer.sum()),
                 spatial=spatial_stats)
    return out, stats


# --------------------------------------------------------------------------- #
# optional helper: depth-weighted fusion of two warped views (NOT paper III-E)
# --------------------------------------------------------------------------- #
def fuse_views(c1, d1, h1, c2, d2, h2, z_eps=1e-3):
    """Depth-weighted blend of two warped views (larger inverse depth = nearer wins).

    This is **not** part of the paper's contribution (the paper renders one virtual view
    from one reference camera); it is provided only for the optional stereo-fusion
    experiment and is clearly marked as such.  Pixels invalid in one view take the other.
    """
    c1 = np.asarray(c1, np.float32)
    c2 = np.asarray(c2, np.float32)
    h1 = as_bool(h1)
    h2 = as_bool(h2)
    w1 = np.where(h1, 0.0, np.asarray(d1, np.float32))
    w2 = np.where(h2, 0.0, np.asarray(d2, np.float32))
    tot = w1 + w2
    out = np.zeros_like(c1)
    both = tot > z_eps
    out[both] = ((c1 * w1[..., None] + c2 * w2[..., None]) / tot[..., None])[both]
    only1 = (~h1) & (h2 | (tot <= z_eps))
    only2 = (~h2) & (h1 | (tot <= z_eps))
    out[only1] = c1[only1]
    out[only2] = c2[only2]
    fused = h1 & h2
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), fused
