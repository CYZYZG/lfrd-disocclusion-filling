"""Reference-guided occlusion-layer prediction (paper III-D extended with a reference search).

Why
---
Diagnostics established three things about the removed band and the disocclusion it feeds:

* stage 5's modified-Criminisi prediction INVENTS the band by extending the surrounding
  content, which is over-smooth exactly where the background has texture
  (tools/texture_energy.py: 0.75 of the ground-truth high-frequency energy survives);
* the sibling reproduction searches the REFERENCE IMAGE for background patches, and on the
  pixels our occlusion layer does not reach it is 3.63 dB ahead
  (tools/coverage_breakdown.py, tools/gap_decompose.py);
* every disocclusion pixel back-projects into the reference frame and ~31% land on reference
  BACKGROUND, where the real texture is available (tools/ref_availability.py).

So instead of inventing the band, search the reference background for it.

How
---
For every pixel of the band:

1. back-project into the reference view with the PREDICTED BACKGROUND DEPTH -- the location of
   the very surface the disocclusion will reveal;
2. within a small window around it, find the displacement minimising the patch SSD between the
   query (the layer predicted so far, which already carries the right geometry) and the
   reference image;
3. accept the reference patch only when the cost is below ``max_cost``, otherwise keep the
   invented value -- a pixel is never worse off than the current behaviour.

The depth-consistency term is what the naive earlier attempts lacked: sampling the reference at
the back-projected position without checking the surface produced 1.09x of the ground-truth
texture energy at the wrong place (tools/texture_oracle.py).

Everything is vectorised over displaced sub-images (the standard cost-volume trick), so the
whole band is processed in one pass per candidate offset.
"""
from __future__ import annotations

import numpy as np


def reference_guided_fill(layer_color, layer_depth, ref_color, pred_depth, region,
                          cams, src, dst, search=6, patch=5, max_cost=48.0,
                          depth_slack=8.0, verbose=False):
    """Return dict(color, depth, replaced_px, accepted_frac, mean_cost, offset).

    The matching cost is computed ONLY on the known (non-removed) pixels of the query patch.
    Matching against the invented part of the layer instead would compare a reference patch with
    a synthesised pattern, which is what the first version did -- it accepted 34% of the band at
    a mean cost of 12.6 but only gained +0.16 dB.  The known neighbours are real reference
    content, so a low cost there means the patch really is the same background surface.
    """
    from . import calib

    H, W = region.shape
    ys, xs = np.nonzero(region)
    n = ys.size
    if n == 0:
        return dict(color=np.array(layer_color, copy=True),
                    depth=np.array(layer_depth, copy=True),
                    replaced_px=0, accepted_frac=0.0, mean_cost=float("nan"),
                    offset=np.zeros((H, W, 2), np.int16))

    # ---- 1. back-projection of the predicted background surface ------------- #
    z = np.asarray(pred_depth, np.float64)[ys, xs]
    u, v, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), z,
                                 cams[dst], cams[src])
    bad = ~np.isfinite(u) | ~np.isfinite(v)
    ur = np.where(bad, -1, np.rint(np.nan_to_num(u))).astype(np.int64)
    vr = np.where(bad, -1, np.rint(np.nan_to_num(v))).astype(np.int64)
    inb = ok & ~bad & (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)
    base_dy = np.where(inb, vr - ys, 0)
    base_dx = np.where(inb, ur - xs, 0)

    lay = np.asarray(layer_color, np.float32)
    ref = np.asarray(ref_color, np.float32)
    dref = np.asarray(pred_depth, np.float64)
    qz = dref[ys, xs]

    # ---- known neighbours of each query patch (real reference content) ------ #
    r = patch // 2
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    oy = np.clip(ys[:, None] + np.array([o[0] for o in offs])[None, :], 0, H - 1)
    ox = np.clip(xs[:, None] + np.array([o[1] for o in offs])[None, :], 0, W - 1)
    known_q = ~region[oy, ox]                          # (n, P) True where the query is real

    best_cost = np.full(n, np.inf)
    best_sy = np.zeros(n, np.int64)
    best_sx = np.zeros(n, np.int64)
    # ---- 2. cost over the local displacement window, on known pixels only --- #
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            sy0 = ys + base_dy + dy
            sx0 = xs + base_dx + dx
            inside = (sy0 - r >= 0) & (sx0 - r >= 0) & (sy0 + r < H) & (sx0 + r < W) & inb
            if not inside.any():
                continue
            sy = np.clip(sy0, 0, H - 1)
            sx = np.clip(sx0, 0, W - 1)
            d_ok = np.abs(dref[sy, sx] - qz) <= depth_slack
            sel = inside & d_ok
            if not sel.any():
                continue
            # reference patch values under the same offsets
            ry = np.clip(sy[:, None] + np.array([o[0] for o in offs])[None, :], 0, H - 1)
            rx = np.clip(sx[:, None] + np.array([o[1] for o in offs])[None, :], 0, W - 1)
            S = ref[ry, rx]                                    # (n, P, 3)
            Q = lay[oy, ox]                                    # (n, P, 3)
            diff = np.abs(S - Q).mean(2)                       # (n, P)
            denom = np.maximum(known_q.sum(1), 1)
            cost = np.where(known_q, diff, 0.0).sum(1) / denom
            better = sel & (cost < best_cost)
            if better.any():
                best_cost[better] = cost[better]
                best_sy[better] = sy[better]
                best_sx[better] = sx[better]

    accept = np.isfinite(best_cost) & (best_cost <= max_cost)
    out_c = np.array(layer_color, copy=True)
    out_d = np.array(layer_depth, copy=True)
    if accept.any():
        ay, ax = ys[accept], xs[accept]
        out_c[ay, ax] = ref[best_sy[accept], best_sx[accept]].astype(np.uint8)
        out_d[ay, ax] = np.clip(pred_depth[best_sy[accept], best_sx[accept]], 0, 255) \
            .astype(np.uint8)
    # remember the accepted displacement for inspection
    off = np.zeros((H, W, 2), np.int16)
    acc = accept
    if acc.any():
        off[ys[acc], xs[acc], 0] = (best_sx[acc] - xs[acc])
        off[ys[acc], xs[acc], 1] = (best_sy[acc] - ys[acc])
    stats = dict(replaced_px=int(accept.sum()),
                 accepted_frac=float(accept.mean()),
                 mean_cost=float(np.nanmean(best_cost[np.isfinite(best_cost)]))
                 if np.isfinite(best_cost).any() else float("nan"))
    if verbose:
        print(f"reference-guided fill: replaced {stats['replaced_px']}/{n} px "
              f"({100 * stats['accepted_frac']:.1f}%), mean best cost {stats['mean_cost']:.1f}")
    return dict(color=out_c, depth=out_d, offset=off, **stats)
