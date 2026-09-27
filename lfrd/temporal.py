"""Temporal background model for the removed region (paper IV/V future work).

The single-view pipeline invents the occluded background.  In a video with a STATIC camera it
can often be *retrieved*: the foreground moves, so a pixel that is occluded in the target frame
shows its true background in other frames.  For MSR Ballet the camera is fixed, so the inverse
depth of a pixel is either the background surface or the moving dancer -> a per-pixel temporal
percentile is a background estimate, and the frames that reach it carry the true colour.

    z_bg(p)   = percentile_t( P_t(p), q )                       (q = 10)
    clipped   = z_bg is pushed down to the level stage 4 measured for that removal run
                (pixels that never become background would otherwise keep the foreground value)
    S(p)      = { t : P_t(p) <= z_bg(p) + tol }
    colour(p) = median_{t in S(p)} I_t(p)                       (only where S is non-empty)

`build_temporal_background` returns that model for the region stage 4 removed.  Where a pixel
has no temporal evidence the caller keeps the single-view prediction, so the two are
complementary rather than exclusive.

Measured on BA54 (cam5->cam4, frames 0-2), compositing through the normal stage 6 machinery:
    single-view occlusion layer   20.84 dB on 39723 px
    + temporal background         22.74 dB on 41120 px      (+1.90 dB, +1397 px covered)
with ~30% of the removed pixels actually becoming background at some frame in the sequence.
"""
from __future__ import annotations

import os

import numpy as np

from . import io_utils


def temporal_depth_percentile(root, cam, frames, region, q=10.0):
    """Per-pixel temporal percentile of the inverse depth on `region`."""
    ys, xs = np.nonzero(region)
    P = np.empty((len(frames), ys.size), np.int16)
    for i, f in enumerate(frames):
        P[i] = io_utils.imread(os.path.join(root, f"cam{cam}",
                                            f"depth-cam{cam}-f{f:03d}.png"),
                               gray=True)[ys, xs]
    return np.percentile(P, q, axis=0).astype(np.float32), P, ys, xs


def build_temporal_background(root, cam, frames, region, q=10.0, tol=4.0,
                              ref_level=None, min_samples=1, agg="mean", agree_tol=6.0):
    """Temporal background colour + depth for the pixels of `region`.

    Parameters
    ----------
    root : dataset root
    cam  : reference camera index
    frames : iterable of frame indices to use (the more the better; the target frame itself
        may be included -- its own foreground pixels simply will not pass the background test)
    region : (H,W) bool, the region to model (stage 4's `removed_mask`)
    q : temporal percentile used as the background depth estimate
    tol : a frame counts as "background" for a pixel when its depth is within tol of z_bg
    ref_level : optional (H,W) float map with the background level stage 4 measured per
        removal run; where the temporal estimate is FARTHER than it, the level is used instead
        (a pixel that never becomes background would otherwise keep its foreground depth)
    min_samples : minimum temporal evidence for a pixel to be part of the model
    agg : how the sampled colours are combined.
        ``"mean"``   : plain average (the original behaviour)
        ``"median"`` : per-pixel median over the samples
        ``"consensus"`` : median, then average only the samples within ``agree_tol`` of that
            median.  The frames a pixel samples are not identical -- they differ by ~8.4 colour
            units on BA54 because the background was partly occluded or moving in some of them
            (tools/temporal_colour_check.py), and a plain average of inconsistent samples is
            how the model produces invented colour at a geometrically correct position
            (tools/shift_on_defects.py measured that shifting the defective blocks back only
            recovers +0.47 dB, i.e. the defect is not a shift).  The consensus average keeps the
            noise reduction of averaging while dropping the outlier frames.
    agree_tol : agreement window for ``agg="consensus"``

    Returns
    -------
    dict(bg_color (H,W,3) uint8, bg_depth (H,W) uint8, n_samples (H,W) int32,
         valid (H,W) bool, clipped (H,W) bool, n_used (H,W) int32)
    """
    ys, xs = np.nonzero(region)
    frames = list(frames)
    n = len(frames)
    P = np.empty((n, ys.size), np.int16)
    COL = np.empty((n, ys.size, 3), np.uint8)
    for i, f in enumerate(frames):
        P[i] = io_utils.imread(os.path.join(root, f"cam{cam}",
                                            f"depth-cam{cam}-f{f:03d}.png"),
                               gray=True)[ys, xs]
        COL[i] = io_utils.imread(os.path.join(root, f"cam{cam}",
                                              f"color-cam{cam}-f{f:03d}.jpg"))[ys, xs]
    z = np.percentile(P, q, axis=0).astype(np.float32)
    clipped = np.zeros(ys.size, bool)
    if ref_level is not None:
        lvl = np.asarray(ref_level, np.float32)[ys, xs]
        ok = np.isfinite(lvl) & (lvl > 0)
        over = ok & (z > lvl + 3.0)
        z[over] = lvl[over]
        clipped = over
    near = (P <= (z[None, :] + tol)) & (P > 0)
    n_s = near.sum(0)
    n_used = n_s.astype(np.int32)
    col = np.zeros((ys.size, 3), np.float32)
    have = n_s > 0
    if agg == "mean":
        acc = np.zeros((ys.size, 3), np.float32)
        for i in range(n):
            sel = near[i]
            if sel.any():
                acc[sel] += COL[i][sel].astype(np.float32)
        col[have] = acc[have] / n_s[have, None]
    else:
        per = np.full((n, ys.size, 3), np.nan, np.float32)
        for i in range(n):
            sel = near[i]
            if sel.any():
                per[i][sel] = COL[i][sel].astype(np.float32)
        with np.errstate(all="ignore"):
            med = np.nanmedian(per, 0)
        have = ~np.isnan(med[:, 0])
        if agg == "median":
            col[have] = med[have]
        else:                                     # consensus
            # keep only the samples that agree with the per-pixel median
            win = np.abs(per - med[None, :, :]) <= agree_tol
            win &= near[..., None]
            cnt = win.sum(0).astype(np.float32)               # (N,3)
            per2 = np.where(win, per, 0.0).sum(0)             # (N,3)
            okk = cnt[:, 0] > 0
            col[okk] = per2[okk] / cnt[okk][:, :1]
            n_used = cnt[:, 0].astype(np.int32)

    H, W = region.shape
    bg_c = np.zeros((H, W, 3), np.uint8)
    bg_d = np.zeros((H, W), np.uint8)
    ns = np.zeros((H, W), np.int32)
    nu = np.zeros((H, W), np.int32)
    valid = np.zeros((H, W), bool)
    clipped_full = np.zeros((H, W), bool)
    sel_ok = have & (n_s >= int(min_samples))
    sl = (ys[sel_ok], xs[sel_ok])
    bg_c[sl] = np.clip(np.rint(col[sel_ok]), 0, 255).astype(np.uint8)
    bg_d[sl] = np.clip(np.rint(z[sel_ok]), 0, 255).astype(np.uint8)
    ns[ys, xs] = n_s
    nu[ys, xs] = n_used
    valid[sl] = True
    clipped_full[ys, xs] = clipped
    return dict(bg_color=bg_c, bg_depth=bg_d, n_samples=ns, n_used=nu, valid=valid,
                clipped=clipped_full)


def apply_to_occlusion_layer(occ_color, occ_depth, model, region):
    """Return copies of the occlusion layer with the temporal model substituted in."""
    c = np.array(occ_color, copy=True)
    d = np.array(occ_depth, copy=True)
    sel = np.asarray(region, bool) & np.asarray(model["valid"], bool)
    c[sel] = model["bg_color"][sel]
    d[sel] = model["bg_depth"][sel]
    return c, d, sel
