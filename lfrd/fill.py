"""Stage B of the reproduction: disocclusion edge classification (paper III-B, eq. 3),
local foreground removal (paper III-C) and removed-region depth prediction
(paper III-D, eq. 4-5).

Where this module sits
----------------------
The virtual view has been synthesised (stage 2, `lfrd/warp.py`) and its holes have been
split into cracks / disocclusions / OOFA.  Here we classify the *disocclusion edge*
pixels as foreground / background, project those edges back into the REFERENCE view and
remove a local piece of the reference foreground that covers the disocclusion.  The
removed region is later (stage 5) filled with background content and warped back into the
virtual view -- that is the whole point of the paper: the hole is never inpainted in the
virtual image.

Conventions (shared with the rest of lfrd/)
------------------------------------------
* colour images are RGB uint8; depth is 8-bit INVERSE depth P (uint8), P == 0 undefined;
* **foreground has a LARGER P** (nearer); background has a smaller P;
* `warped_depth` is float/int with -1 marking a hole (no warped sample);
* masks are uint8 with 255 = hole / True;
* `u` is the horizontal image coordinate (x, column), `v` the vertical one (y, row).

Deviations / interpretations (paper has no code; see the run report)
--------------------------------------------------------------------
1.  eq. (3) is evaluated on the warped Laplacian of the pixel just *outside* the
    disocclusion, because a hole pixel has no warped Laplacian of its own; the strongest
    (max |value|) sample within `edge_band` px is used.
2.  The cross-check ("the side with the larger depth is foreground") compares the depth
    of the outward band with the depth the disocclusion *reveals*, sampled ACROSS the hole
    (`_cross_hole_depth`: for a left/right edge pixel, the first valid pixel on the other
    side of the hole in that row).  Following the spec the depth rule wins when it is
    determinate, otherwise eq. (3) is used; on BA54 f000 the two rules agree on 95.2% of
    the edge pixels.
3.  The reference edge position (paper III-C step 1) is taken from stage 2's backward
    index of the valid pixel just outside the disocclusion on the background side: that is
    the exact reference source of the content displayed there, i.e. the first background
    pixel behind the occluder.  Only when no backward index exists is the boundary pixel
    inverse-projected (with the nearest-valid depth, converted from inverse depth P to the
    distance z).  The removal band then extends from that reference position towards the
    occluder by the disocclusion width of the row ("the local foreground region is removed
    based on the width of disocclusion").
    The narrower alternative -- only the interval between the two inverse-projected edge
    positions -- was measured on BA54 f000: it removes 3547 px and the re-warped layer
    covers only 0.12 of the disocclusions, whereas the band removes 50390 px and covers
    0.92 (area weighted).  A disocclusion contains no visible content by construction, so
    the layer that fills it must be as wide as the hole; the narrow interval leaves the
    hole empty.
4.  The depth of the region a disocclusion reveals is measured locally (the median
    outward depth of its BG-classified edge pixels) and used as the single foreground
    definition (`P_ref > bg_depth + margin`) for the removal, for the inverse projection
    and for the eq.(4) endpoint classification.  A window-based global mask cannot be used
    here: a local minimum over a 21x21 window inside a large foreground object returns the
    object's own depth, so it marks the object interior as background.
5.  With the footprint above, the reference band width equals the disocclusion width mapped
    to reference coordinates, so paper case 1 ("remove a band as wide as the disocclusion",
    anchored on the background side) and case 2 ("remove all foreground pixels between the
    two reference edges") coincide row by row for BA54; both are implemented literally and
    `full_obj_in_case_b=True` offers the stronger reading of case 2.
"""
import os
from collections import Counter, OrderedDict

import cv2
import numpy as np
from scipy import ndimage

from . import calib, io_utils

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _bool(a):
    return np.asarray(a) > 0


def scale_cams_to_shape(cams, shape):
    """Adapt a calibration to an image of a different resolution (fixture support).

    `calib.project/_unproject` hard-code the 1024x768 principal-point convention, so a
    down-scaled image needs a K that keeps the same camera ray.  For the real 1024x768
    data this is a no-op.
    """
    H, W = int(shape[0]), int(shape[1])
    if (H, W) == (calib.IMG_H, calib.IMG_W):
        return cams
    sy, sx = H / float(calib.IMG_H), W / float(calib.IMG_W)
    out = {}
    for k, c in cams.items():
        K = np.array(c["K"], dtype=np.float64)
        cy_src = (calib.IMG_H - 1) - K[1, 2]           # principal point, top-down
        cx_src = K[0, 2]
        K[0, 0] *= sx
        K[0, 1] *= sx
        K[0, 2] = cx_src * sx
        K[1, 0] *= sy
        K[1, 1] *= sy
        K[1, 2] = (calib.IMG_H - 1) - cy_src * sy
        c2 = dict(c)
        c2["K"] = K
        out[k] = c2
    return out


# --------------------------------------------------------------------------- #
# depth fill
# --------------------------------------------------------------------------- #
def fill_hole_nearest(depth, valid=None, hole=None, fill_value=0.0):
    """Replace hole pixels by the value of the nearest valid pixel.

    `valid` (True = usable) defaults to `depth >= 0` (the warped-depth convention) unless
    an explicit `hole` mask is given.  Uses `scipy.ndimage.distance_transform_edt`.
    """
    d = np.asarray(depth, dtype=np.float64)
    if valid is None:
        valid = (~_bool(hole)) if hole is not None else (d >= 0)
    valid = _bool(valid)
    hole = (~valid) if hole is None else _bool(hole)
    out = d.copy()
    if not valid.any():
        return np.full_like(out, float(fill_value))
    if hole.any():
        _dist, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
        h = hole
        out[h] = d[iy[h], ix[h]]
    return out


def fill_hole_background(depth, hole, seed, depth_valid=None):
    """Fill `hole` pixels with the depth of the nearest pixel of `seed`.

    `seed` is the set of valid pixels believed to be the background the holes reveal
    (stage 2 convention: the far surface, i.e. the *smaller* P).  Falls back to
    `fill_hole_nearest` when the seed is empty.
    """
    d = np.asarray(depth, dtype=np.float64)
    hole = _bool(hole)
    seed = _bool(seed)
    if depth_valid is not None:
        seed = seed & _bool(depth_valid)
    if not seed.any():
        return fill_hole_nearest(d, hole=hole)
    out = d.copy()
    if hole.any():
        _dist, (iy, ix) = ndimage.distance_transform_edt(~seed, return_indices=True)
        out[hole] = d[iy[hole], ix[hole]]
    return out


def reference_foreground_mask(P, win=21, margin=6.0):
    """Near-object mask of a reference inverse-depth image (foreground = larger P).

    The local background level is the minimum of P in a `win` x `win` window (the
    farthest surface around the pixel, separable min filter).  `margin` is in P units.
    """
    P = np.asarray(P, dtype=np.float32)
    win = int(win)
    bg = cv2.erode(P, np.ones((1, win), np.uint8))
    bg = cv2.erode(bg, np.ones((win, 1), np.uint8))
    return (P > (bg + float(margin)))


# --------------------------------------------------------------------------- #
# stage-2 artefact loading + edges.npz round trip (shared by step3 and step4)
# --------------------------------------------------------------------------- #
def load_warp_artefacts(warp_dir):
    """Load the `20_warp` artefacts needed by steps 3-4.

    Returns dict(warped_color RGB, warped_depth float32 (-1 = hole), warped_lap float32,
                 hole_disocc / hole_all / hole_crack / hole_oofa bool,
                 backward int32 flat source index or None, dir).
    """
    p = lambda n: os.path.join(warp_dir, n)                        # noqa: E731
    out = {"dir": warp_dir}
    out["warped_color"] = io_utils.imread(p("warped_color.png"), gray=False)
    if not os.path.isfile(p("warped_depth.npy")):
        raise FileNotFoundError(f"missing {p('warped_depth.npy')} (run step2 first)")
    out["warped_depth"] = np.load(p("warped_depth.npy")).astype(np.float32)
    out["warped_lap"] = (np.load(p("warped_lap.npy")).astype(np.float32)
                         if os.path.isfile(p("warped_lap.npy"))
                         else np.zeros_like(out["warped_depth"]))
    gray = lambda n: (io_utils.imread(p(n), gray=True) > 0)         # noqa: E731
    for name, fname in (("hole_all", "hole_all.png"), ("hole_crack", "hole_crack.png"),
                        ("hole_oofa", "hole_oofa.png"), ("hole_disocc", "hole_disocc.png")):
        if os.path.isfile(p(fname)):
            out[name] = gray(fname)
    if "hole_disocc" not in out:
        if "hole_all" not in out:
            raise FileNotFoundError(f"neither hole_disocc.png nor hole_all.png in {warp_dir}")
        d = out["hole_all"].copy()
        for k in ("hole_crack", "hole_oofa"):
            if k in out:
                d &= ~out[k]
        out["hole_disocc"] = d
        out["hole_disocc_derived"] = True
    out.setdefault("hole_all", out["hole_disocc"].copy())
    out["backward"] = None
    if os.path.isfile(p("backward.npz")):
        for k in ("idx", "backward", "index"):
            try:
                d = io_utils.load_npz(p("backward.npz"))
            except Exception:
                break
            if k in d:
                out["backward"] = d[k].astype(np.int32)
                break
        if out["backward"] is None:
            d = io_utils.load_npz(p("backward.npz"))
            out["backward"] = np.asarray(list(d.values())[0]).astype(np.int32)
    return out


def save_edges_npz(path, edges):
    """Serialise a `classify_edges` result to a flat npz (variable length lists)."""
    comps = edges["components"]
    n = len(comps)
    ptr = [0]
    by, bx, F, Fl, Fd = [], [], [], [], []
    for c in comps:
        by.append(c["boundary_y"]); bx.append(c["boundary_x"])
        F.append(c["F"]); Fl.append(c["F_lap"]); Fd.append(c["F_dep"])
        ptr.append(ptr[-1] + int(c["n_boundary"]))
    cat = lambda a, dt: (np.concatenate(a).astype(dt) if a else np.zeros(0, dt))  # noqa: E731
    io_utils.save_npz(
        path,
        labels=np.asarray(edges["labels"], np.int32),
        fg_edge=np.asarray(edges["fg_edge"], np.uint8),
        bg_edge=np.asarray(edges["bg_edge"], np.uint8),
        n_comp=np.int32(n),
        comp_id=np.array([c["id"] for c in comps], np.int32),
        comp_bbox=np.array([c["bbox"] for c in comps], np.int32).reshape(n, 4),
        comp_area=np.array([c["area"] for c in comps], np.int32),
        comp_thickness=np.array([c["thickness"] for c in comps], np.float32),
        comp_bg_side=np.array([c["bg_side"] for c in comps], np.int8),
        comp_centroid=np.array([c["centroid"] for c in comps], np.float32).reshape(n, 2),
        comp_bg_depth=np.array([c["bg_depth"] for c in comps], np.float32),
        comp_fg_depth=np.array([c["fg_depth"] for c in comps], np.float32),
        comp_mean_F=np.array([c["mean_depth_F"] for c in comps], np.float32),
        comp_mean_BG=np.array([c["mean_depth_BG"] for c in comps], np.float32),
        comp_agree=np.array([c["agree_rate"] for c in comps], np.float32),
        comp_n_agree=int(sum(c["n_agreed"] for c in comps)),
        comp_n_agreeable=int(sum(c["n_agreeable"] for c in comps)),
        comp_n_undet=int(sum(c["n_undetermined"] for c in comps)),
        stat_mean_F=np.float32(edges["stats"]["mean_depth_F"]),
        stat_mean_BG=np.float32(edges["stats"]["mean_depth_BG"]),
        stat_boundary=np.int64(edges["stats"]["n_boundary_pixels"]),
        comp_nboundary=np.array([c["n_boundary"] for c in comps], np.int32),
        bnd_ptr=np.array(ptr, np.int64),
        bnd_y=cat(by, np.int32), bnd_x=cat(bx, np.int32),
        bnd_F=cat(F, np.int8), bnd_Flap=cat(Fl, np.int8), bnd_Fdep=cat(Fd, np.int8),
    )
    return path


def load_edges_npz(path):
    """Inverse of `save_edges_npz` -> dict(labels, components, fg_edge, bg_edge, stats).

    `ref_rows` / `u_ref` are NOT restored: step4 recomputes them with
    `inverse_project_edges` (they depend on the camera pair).
    """
    d = io_utils.load_npz(path)
    n = int(d["n_comp"])
    ptr = d["bnd_ptr"]
    comps = []
    for i in range(n):
        a, b = int(ptr[i]), int(ptr[i + 1])
        comps.append(dict(
            id=int(d["comp_id"][i]),
            bbox=tuple(int(v) for v in d["comp_bbox"][i]),
            area=int(d["comp_area"][i]), thickness=float(d["comp_thickness"][i]),
            bg_side=int(d["comp_bg_side"][i]),
            centroid=tuple(float(v) for v in d["comp_centroid"][i]),
            bg_depth=float(d["comp_bg_depth"][i]), fg_depth=float(d["comp_fg_depth"][i]),
            mean_depth_F=float(d["comp_mean_F"][i]),
            mean_depth_BG=float(d["comp_mean_BG"][i]),
            agree_rate=float(d["comp_agree"][i]),
            n_boundary=b - a, boundary_y=d["bnd_y"][a:b].astype(np.int32),
            boundary_x=d["bnd_x"][a:b].astype(np.int32),
            F=d["bnd_F"][a:b].astype(np.int8), F_lap=d["bnd_Flap"][a:b].astype(np.int8),
            F_dep=d["bnd_Fdep"][a:b].astype(np.int8),
            n_agreed=0, n_agreeable=0, n_undetermined=0,
        ))
    agg = int(d["comp_n_agree"]); agb = int(d["comp_n_agreeable"])
    stats = dict(n_components=n,
                 n_boundary_pixels=int(d["stat_boundary"]),
                 n_undetermined_pixels=int(d["comp_n_undet"]),
                 agree_n=agg, agree_n_able=agb,
                 agree_rate=(agg / float(agb)) if agb else float("nan"),
                 mean_depth_F=float(d["stat_mean_F"]),
                 mean_depth_BG=float(d["stat_mean_BG"]),
                 bg_side_counts=dict(Counter(int(v) for v in d["comp_bg_side"])))
    return dict(labels=d["labels"], comp_ids=[c["id"] for c in comps], components=comps,
                fg_edge=d["fg_edge"], bg_edge=d["bg_edge"], stats=stats)



def _outward_band_stats(mask, by, bx, lap, depth, valid, edge_band):
    """For every boundary pixel, sample the band of valid pixels just OUTSIDE `mask`.

    Returns (mean outward depth, strongest-|Laplacian| outward sample, outward count).
    Vectorised over the (2*edge_band+1)^2 - 1 offsets.
    """
    H, W = mask.shape
    b = int(edge_band)
    nb = by.size
    dsum = np.zeros(nb, np.float64)
    cnt = np.zeros(nb, np.int32)
    best = np.zeros(nb, np.float32)
    best_abs = np.zeros(nb, np.float32)
    for dy in range(-b, b + 1):
        for dx in range(-b, b + 1):
            if dy == 0 and dx == 0:
                continue
            ny = by + dy
            nx = bx + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            if not ok.any():
                continue
            k = np.nonzero(ok)[0]
            yy, xx = ny[k], nx[k]
            sel = (~mask[yy, xx]) & valid[yy, xx]
            if not sel.any():
                continue
            k = k[sel]
            yy, xx = yy[sel], xx[sel]
            dsum[k] += depth[yy, xx]
            cnt[k] += 1
            lv = lap[yy, xx]
            al = np.abs(lv)
            upd = al > best_abs[k]
            if upd.any():
                kk = k[upd]
                best[kk] = lv[upd]
                best_abs[kk] = al[upd]
    d_out = np.where(cnt > 0, dsum / np.maximum(cnt, 1), np.nan)
    return d_out, best, cnt


def _cross_hole_depth(mask, by, bx, depth):
    """Depth of the surface the hole reveals, sampled ACROSS the hole.

    For a boundary pixel on the left/right edge of the component, the sample is the first
    valid pixel on the other side of the hole in that row; for a top/bottom edge pixel it
    is the first valid pixel above/below the component in that column.  This is the
    "inward band" of the spec: a disocclusion reveals background, so the surface across
    the hole is the background the hole covers.  NaN where nothing valid is found.
    """
    H, W = mask.shape
    valid = depth > 0
    ys_all, xs_all = np.nonzero(mask)
    rows_span, cols_span = {}, {}
    if ys_all.size:
        order = np.argsort(ys_all, kind="stable")
        yy, xx = ys_all[order], xs_all[order]
        uy, st = np.unique(yy, return_index=True)
        lo = np.minimum.reduceat(xx, st)
        hi = np.maximum.reduceat(xx, st)
        rows_span = {int(y): (int(a), int(b)) for y, a, b in zip(uy, lo, hi)}
        order = np.argsort(xs_all, kind="stable")
        xx2, yy2 = xs_all[order], ys_all[order]
        ux, st = np.unique(xx2, return_index=True)
        lo = np.minimum.reduceat(yy2, st)
        hi = np.maximum.reduceat(yy2, st)
        cols_span = {int(x): (int(a), int(b)) for x, a, b in zip(ux, lo, hi)}

    row_valid, col_valid = {}, {}

    def _rv(y):
        if y not in row_valid:
            row_valid[y] = np.nonzero(valid[y])[0]
        return row_valid[y]

    def _cv(x):
        if x not in col_valid:
            col_valid[x] = np.nonzero(valid[:, x])[0]
        return col_valid[x]

    d_in = np.full(by.size, np.nan)
    for j in range(by.size):
        y, x = int(by[j]), int(bx[j])
        rs = rows_span.get(y)
        cs = cols_span.get(x)
        if rs is not None and x <= rs[0]:
            vi = _rv(y)
            k = int(np.searchsorted(vi, rs[1], side="right"))
            if k < vi.size:
                d_in[j] = depth[y, vi[k]]
        elif rs is not None and x >= rs[1]:
            vi = _rv(y)
            k = int(np.searchsorted(vi, rs[0], side="left")) - 1
            if k >= 0:
                d_in[j] = depth[y, vi[k]]
        elif cs is not None and y <= cs[0]:
            vj = _cv(x)
            k = int(np.searchsorted(vj, cs[1], side="right"))
            if k < vj.size:
                d_in[j] = depth[vj[k], x]
        elif cs is not None and y >= cs[1]:
            vj = _cv(x)
            k = int(np.searchsorted(vj, cs[0], side="left")) - 1
            if k >= 0:
                d_in[j] = depth[vj[k], x]
    return d_in


def _vote_background_side(by, bx, F, centroid_x, d_out):
    """Which side of the component is the background: +1 right, -1 left.

    Primary vote: per row, the component's disocclusion edge must be FG-BG (background
    right, +1) or BG-FG (background left, -1).  Fallbacks: mean x of the BG side pixels,
    then the depth comparison (the background side has the smaller depth).
    """
    votes, n_votes = 0, 0
    for y in np.unique(by):
        sel = by == y
        xr = bx[sel]
        Fr = F[sel]
        if xr.size < 2:
            continue
        il = int(np.argmin(xr))
        ir = int(np.argmax(xr))
        if il == ir:
            continue
        if Fr[il] == 1 and Fr[ir] == 0:
            votes += 1
            n_votes += 1
        elif Fr[il] == 0 and Fr[ir] == 1:
            votes -= 1
            n_votes += 1
    if votes != 0:
        return (1 if votes > 0 else -1), "row-vote", n_votes
    bgpix = F == 0
    if bgpix.any():
        return (1 if bx[bgpix].mean() >= centroid_x else -1), "bg-mean-x", n_votes
    left = np.nanmean(d_out[bx <= centroid_x]) if (bx <= centroid_x).any() else np.nan
    right = np.nanmean(d_out[bx > centroid_x]) if (bx > centroid_x).any() else np.nan
    if np.isfinite(left) and np.isfinite(right) and left != right:
        return (1 if right < left else -1), "depth", n_votes
    return 1, "default", n_votes


def classify_edges(hole_disocc, warped_lap, warped_depth, edge_band=2, use_laplacian=True,
                   min_area=1, min_width=0, bg_margin=6.0, lap_eps=0.0):
    """Classify the disocclusion edge pixels into foreground / background.

    Parameters
    ----------
    hole_disocc : (H,W) uint8/bool   disocclusion mask (255/bool = hole)
    warped_lap  : (H,W) float        warped Laplacian of the preprocessed depth
    warped_depth: (H,W)              warped inverse depth, -1 where the pixel is a hole
    edge_band   : half width (px) of the bands used for the eq.(3) / depth cross-check
    use_laplacian : False -> depth-only ablation
    min_area, min_width : ignore components smaller than this (cfg.holes_min_area / width)

    Returns
    -------
    dict with keys
      labels      (H,W) int32   component label image of hole_disocc (0 = no hole)
      comp_ids    list[int]
      components  list[dict]    per component: id, bbox (x,y,w,h), area, thickness,
                                boundary_y/x, F (0/1), F_lap, F_dep (-1 unknown),
                                bg_side (+1/-1), bg_side_rule, agree_rate,
                                mean_depth_F, mean_depth_BG, bg_depth, fg_depth,
                                centroid (cx,cy), n_boundary, n_agree
      fg_edge, bg_edge  (H,W) uint8  255 at the F=1 / F=0 boundary pixels
      stats       dict          totals + eq.(4)-style counts for the report
    """
    hole = _bool(hole_disocc)
    lap = np.asarray(warped_lap, dtype=np.float32)
    depth = np.asarray(warped_depth, dtype=np.float32)
    if hole.shape != lap.shape or hole.shape != depth.shape:
        raise ValueError(f"shape mismatch: hole {hole.shape}, lap {lap.shape}, "
                         f"depth {depth.shape}")
    H, W = hole.shape
    valid = depth >= 0
    depth_nn = fill_hole_nearest(depth, valid=valid)

    n_lab, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    fg_edge = np.zeros((H, W), np.uint8)
    bg_edge = np.zeros((H, W), np.uint8)
    comps = []
    tot_boundary = tot_undet = tot_agree = tot_agreeable = 0
    fg_depths, bg_depths = [], []

    for i in range(1, n_lab):
        m = labels == i
        area = int(m.sum())
        if area < int(min_area):
            continue
        ys, xs = np.nonzero(m)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        thick = float(cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3).max())
        width_est = 2.0 * thick
        if width_est < float(min_width):
            continue

        er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
        bnd = m & (er == 0)
        by, bx = np.nonzero(bnd)
        nb = by.size

        # ---- outward ring of the component (all offsets up to edge_band) ---- #
        k = 2 * int(edge_band) + 1
        ring = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        ring = ring & (~m) & valid
        if ring.any():
            rv = depth[ring]
            bg_depth = float(np.percentile(rv, 10))
            fg_depth = float(np.percentile(rv, 90))
        else:
            bg_depth = fg_depth = float("nan")

        # ---- eq.(3) + depth cross-check on every boundary pixel ------------- #
        d_out, lap_best, out_cnt = _outward_band_stats(m, by, bx, lap, depth, valid,
                                                       edge_band)
        d_in = _cross_hole_depth(m, by, bx, depth)
        F_lap = np.full(nb, -1, np.int8)
        det_lap = lap_best < -float(lap_eps)
        F_lap[det_lap] = 1
        det_lap0 = lap_best > float(lap_eps)
        F_lap[det_lap0] = 0

        F_dep = np.full(nb, -1, np.int8)
        both_valid = np.isfinite(d_out) & np.isfinite(d_in)
        F_dep[both_valid & (d_out > d_in + float(bg_margin))] = 1
        F_dep[both_valid & (d_out < d_in - float(bg_margin))] = 0

        if use_laplacian:
            F = np.where(F_dep >= 0, F_dep, np.where(F_lap >= 0, F_lap, 0)).astype(np.int8)
        else:
            F = np.where(F_dep >= 0, F_dep, 0).astype(np.int8)

        both = (F_lap >= 0) & (F_dep >= 0)
        n_agree = int((F_lap[both] == F_dep[both]).sum()) if both.any() else 0
        agree_rate = (n_agree / float(both.sum())) if both.any() else float("nan")
        n_undet = int((F_lap < 0).sum()) + int((F_dep < 0).sum())

        fg_sel = F == 1
        bg_sel = F == 0
        fg_edge[by[fg_sel], bx[fg_sel]] = 255
        bg_edge[by[bg_sel], bx[bg_sel]] = 255
        dnn = depth_nn[by, bx]
        mean_f = float(dnn[fg_sel].mean()) if fg_sel.any() else float("nan")
        mean_b = float(dnn[bg_sel].mean()) if bg_sel.any() else float("nan")
        # locally measured (revealed) background / foreground depth from the outward band
        ring_bg, ring_fg = bg_depth, fg_depth
        bg_loc = (float(np.nanmedian(d_out[bg_sel]))
                  if bg_sel.any() and np.isfinite(d_out[bg_sel]).any() else ring_bg)
        fg_loc = (float(np.nanmedian(d_out[fg_sel]))
                  if fg_sel.any() and np.isfinite(d_out[fg_sel]).any() else ring_fg)
        if not np.isfinite(bg_loc):
            bg_loc = ring_bg
        if not np.isfinite(fg_loc):
            fg_loc = ring_fg
        # if the two measurements are inverted (mis-classification), keep the far/near pair
        if np.isfinite(bg_loc) and np.isfinite(fg_loc) and bg_loc >= fg_loc:
            cands = [v for v in (bg_loc, fg_loc, ring_bg) if np.isfinite(v)]
            cands_f = [v for v in (bg_loc, fg_loc, ring_fg) if np.isfinite(v)]
            bg_loc, fg_loc = float(min(cands)), float(max(cands_f))
        fg_depths.append(dnn[fg_sel])
        bg_depths.append(dnn[bg_sel])

        cx, cy = float(xs.mean()), float(ys.mean())
        bg_side, side_rule, n_votes = _vote_background_side(by, bx, F, cx, d_out)

        comps.append(dict(
            id=int(i), bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1), area=area,
            thickness=thick, width_est=width_est,
            boundary_y=by.astype(np.int32), boundary_x=bx.astype(np.int32),
            F=F, F_lap=F_lap, F_dep=F_dep,
            n_boundary=int(nb), n_agreed=n_agree, n_agreeable=int(both.sum()),
            n_undetermined=n_undet, agree_rate=agree_rate,
            mean_depth_F=mean_f, mean_depth_BG=mean_b,
            bg_depth=bg_depth, fg_depth=fg_depth,
            centroid=(cx, cy), bg_side=int(bg_side), bg_side_rule=side_rule,
            n_side_votes=int(n_votes),
        ))
        tot_boundary += nb
        tot_undet += n_undet
        tot_agree += n_agree
        tot_agreeable += int(both.sum())

    fg_all = np.concatenate(fg_depths) if fg_depths else np.zeros(0)
    bg_all = np.concatenate(bg_depths) if bg_depths else np.zeros(0)
    stats = dict(
        n_components=len(comps),
        n_boundary_pixels=int(tot_boundary),
        n_undetermined_pixels=int(tot_undet),
        agree_n=int(tot_agree), agree_n_able=int(tot_agreeable),
        agree_rate=(tot_agree / float(tot_agreeable)) if tot_agreeable else float("nan"),
        mean_depth_F=float(fg_all.mean()) if fg_all.size else float("nan"),
        mean_depth_BG=float(bg_all.mean()) if bg_all.size else float("nan"),
        bg_side_counts=dict(Counter(int(c["bg_side"]) for c in comps)),
    )
    return dict(labels=labels, comp_ids=[c["id"] for c in comps], components=comps,
                fg_edge=fg_edge, bg_edge=bg_edge, stats=stats)


# --------------------------------------------------------------------------- #
# 2. inverse projection of the disocclusion edges into the reference view
# --------------------------------------------------------------------------- #
def _source_anchor(backward, P, depth_nn, cams, src, dst, x, y, x_step, max_walk=8,
                   prefer_max=None):
    """Reference-view (u, v) of the content displayed at virtual pixel (x, y).

    Uses stage 2's flat backward index when it is valid (`>= 0`) -- that is the exact
    reference source pixel of the content, so no depth is needed.  If the pixel at (x, y)
    has no source (it is itself a hole) the walk continues in `x_step` direction for at
    most `max_walk` px.  `prefer_max`: walk on until the source depth is <= this value
    (used for the background side, so that a thin foreground sliver next to the hole does
    not become the anchor).  Falls back to the inverse projection with the nearest-valid
    depth when no backward index is available at all.
    """
    H, W = depth_nn.shape
    first = None
    if backward is not None:
        xx = int(x)
        for _ in range(int(max_walk)):
            if not (0 <= xx < W and 0 <= y < H):
                break
            k = int(backward[y * W + xx])
            if k >= 0:
                uu, vv = k % W, k // W
                if prefer_max is None or not np.isfinite(prefer_max):
                    return (uu, vv)
                if first is None:
                    first = (uu, vv)
                if P is not None and P[vv, uu] <= prefer_max:
                    return (uu, vv)
            xx += int(x_step)
    if 0 <= x < W and 0 <= y < H:
        z = float(calib.depth_from_P(np.array([depth_nn[y, x]]))[0])
        uu, vv, ok = calib.project_pts(np.float64(x), np.float64(y), np.float64(z),
                                       cams[dst], cams[src])
        if np.isfinite(uu) and np.isfinite(vv):
            return (int(round(float(uu))), int(round(float(vv))))
    return first


def inverse_project_edges(edges, warped_depth, cams, src, dst, backward_idx=None,
                          mode="nearest", bg_margin=6.0, P_ref=None):
    """Map every disocclusion back into the REFERENCE view (paper III-C step 1).

    For each component two things are produced:

    * `u_ref`, `v_ref`: the projection of the component's pixels back into the reference
      view with the warped depth (nearest-valid fill, as the spec/paper prescribe -- the
      `mode` argument switches to a background-depth fill as an ablation).  This is the
      literal "projection of the component".
    * `ref_rows`, `footprint`: the per-row **reference exposure band**, which is what the
      removal uses.  The background-side disocclusion edge pixels are projected with the
      depth of the background they are adjacent to (nearest valid pixel, distance 1);
      that gives the reference position of the background-side edge.  From there the band
      extends towards the opposite side by the disocclusion width of that row -- paper
      III-C: "the local foreground region is removed based on the width of disocclusion".
      Projecting the *whole* hole would need a depth for pixels that have none, and a
      single deepest-background fill collapses the footprint onto one point, so the
      paper's edge-pixel projection plus the measured hole width is used instead.

    Note: `warped_depth` stores the warped INVERSE depth P (0..255, -1 = hole); every
    projection converts it with `calib.depth_from_P` because `project_pts` expects the
    distance z.

    `backward_idx` (flat source index per virtual pixel, -1 where empty, from stage 2) is
    preferred whenever it is valid, as the spec prescribes -- for a true disocclusion it
    is -1 and the projection path above is used.

    Returns dict(components=[... with u_ref, v_ref, row_span, ref_rows, ref_opp, fp_bbox],
                 footprint (H,W) uint8 = exposure bands, footprint_label (H,W) int32,
                 proj_mask (H,W) uint8 = projection of the component pixels, stats).
    """
    depth = np.asarray(warped_depth, dtype=np.float32)
    valid = depth >= 0
    H, W = depth.shape
    lab = np.asarray(edges["labels"])
    if lab.shape != depth.shape:
        raise ValueError(f"labels {lab.shape} vs depth {depth.shape}")
    depth_nn = fill_hole_nearest(depth, valid=valid)
    if backward_idx is not None:
        backward_idx = np.asarray(backward_idx).reshape(-1)
    if P_ref is not None:
        P_ref = np.asarray(P_ref, dtype=np.float32)
        if P_ref.shape != depth.shape:
            raise ValueError(f"P_ref {P_ref.shape} vs warped depth {depth.shape}")

    fp_label = np.zeros((H, W), np.int32)
    proj_mask = np.zeros((H, W), np.uint8)
    out_comps = []
    n_missing_bb = 0

    for c in edges["components"]:
        i = int(c["id"])
        m = lab == i
        ys, xs = np.nonzero(m)
        z = calib.depth_from_P(depth_nn[ys, xs])
        if mode == "background" and np.isfinite(c.get("bg_depth", np.nan)):
            seed = valid & (depth <= float(c["bg_depth"]) + float(bg_margin))
            zf = fill_hole_background(depth, m, seed, depth_valid=valid)
            z = calib.depth_from_P(zf[ys, xs])

        uref, vref, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64),
                                           z.astype(np.float64), cams[dst], cams[src])
        uref = np.asarray(uref, dtype=np.float64)
        vref = np.asarray(vref, dtype=np.float64)
        if backward_idx is not None:
            flat = ys.astype(np.int64) * W + xs.astype(np.int64)
            has = backward_idx[flat] >= 0
            if has.any():
                su = backward_idx[flat[has]] % W
                sv = backward_idx[flat[has]] // W
                uref[has] = su
                vref[has] = sv

        ur = np.rint(uref).astype(np.int64)
        vr = np.rint(vref).astype(np.int64)
        inb = (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)
        if inb.any():
            proj_mask[vr[inb], ur[inb]] = 255

        # ---- per-row hole span of the component --------------------------- #
        order = np.argsort(ys, kind="stable")
        yy, xx = ys[order], xs[order]
        uy, st = np.unique(yy, return_index=True)
        h_lo = np.minimum.reduceat(xx, st)
        h_hi = np.maximum.reduceat(xx, st)
        row_span = {int(y): (int(a), int(b)) for y, a, b in zip(uy, h_lo, h_hi)}

        # ---- reference edge anchors (paper III-C step 1) ------------------- #
        # The BACKGROUND-side anchor is the exact reference source of the valid content
        # displayed just outside the hole on the background side (stage 2 backward map),
        # i.e. the reference position of the first background pixel behind the occluder.
        # The band then extends from there towards the occluder by the disocclusion width
        # of that row ("the local foreground region is removed based on the width of
        # disocclusion").  Both anchors must be reference positions, hence the projection
        # fallback converts the warped inverse depth P to the distance z.
        s = int(c["bg_side"])
        bg_lvl = float(c.get("bg_depth", np.nan))
        by, bx, F = c["boundary_y"], c["boundary_x"], c["F"]
        ref_rows = OrderedDict()
        ref_opp = {}
        for y in np.unique(ys):
            span = row_span.get(int(y))
            if span is None:
                continue
            hxl, hxr = span
            wr = hxr - hxl + 1                       # disocclusion width on this row
            x_bg = (hxr + 1) if s > 0 else (hxl - 1)
            x_op = (hxl - 1) if s > 0 else (hxr + 1)
            a_bg = _source_anchor(backward_idx, P_ref, depth_nn, cams, src, dst, x_bg,
                                  int(y), 1 if s > 0 else -1,
                                  prefer_max=(bg_lvl + float(bg_margin)
                                              if np.isfinite(bg_lvl) else None))
            a_op = _source_anchor(backward_idx, P_ref, depth_nn, cams, src, dst, x_op,
                                  int(y), -1 if s > 0 else 1)
            if a_bg is None:
                continue
            u_a, v_a = int(a_bg[0]), int(a_bg[1])
            if s > 0:
                lo, hi = u_a - wr + 1, u_a
            else:
                lo, hi = u_a, u_a + wr - 1
            lo, hi = max(0, lo), min(W - 1, hi)
            if hi < lo:
                continue
            cur = ref_rows.get(v_a)
            ref_rows[v_a] = (lo, hi) if cur is None else (min(cur[0], lo),
                                                          max(cur[1], hi))
            if a_op is not None:
                ref_opp[v_a] = int(a_op[0])

        for y, (lo, hi) in ref_rows.items():
            fp_label[y, lo:hi + 1] = i
        if ref_rows:
            vv = np.array(list(ref_rows.keys()))
            lo = min(v[0] for v in ref_rows.values())
            hi = max(v[1] for v in ref_rows.values())
            fp_bbox = (int(lo), int(vv.min()), int(hi), int(vv.max()))
        else:
            fp_bbox = None
            n_missing_bb += 1

        c2 = dict(c)
        c2.update(u_ref=uref, v_ref=vref, n_proj=int(inb.sum()), row_span=row_span,
                  ref_rows=ref_rows, ref_opp=ref_opp, fp_bbox=fp_bbox)
        out_comps.append(c2)

    stats = dict(n_components=len(out_comps), n_no_footprint=n_missing_bb,
                 n_footprint_px=int((fp_label > 0).sum()), mode=mode)
    return dict(components=out_comps, footprint=(fp_label > 0).astype(np.uint8),
                footprint_label=fp_label, proj_mask=proj_mask, labels=lab, stats=stats)


# --------------------------------------------------------------------------- #
# 3. local foreground removal  (paper III-C)
# --------------------------------------------------------------------------- #
def removal_case(c, border=0):
    """Case of paper III-C from the classification opposite to the background side.

    Returns ("A"|"B", n_votes, n_rows): "A" when the edge opposite to the background side
    is foreground (only part of the foreground is exposed -> remove a band as wide as the
    disocclusion); "B" when it is background (the whole foreground was warped away).
    """
    s = int(c["bg_side"])
    by, bx, F = c["boundary_y"], c["boundary_x"], c["F"]
    A = B = 0
    for y in np.unique(by):
        sel = by == y
        xr = bx[sel]
        Fr = F[sel]
        if xr.size < 2:
            continue
        i_opp = int(np.argmin(xr)) if s > 0 else int(np.argmax(xr))
        if Fr[i_opp] == 1:
            A += 1
        else:
            B += 1
    if A == 0 and B == 0:
        return "A", 0, 0
    case = "A" if A >= B else "B"
    return case, (A if case == "A" else B), A + B


def remove_local_foreground(P, fg_mask, holes, cams, src, dst, cfg,
                            bg_margin=6.0, full_obj_in_case_b=False,
                            min_removal_area=100):
    """Remove the local reference foreground that covers each disocclusion (paper III-C).

    Parameters
    ----------
    P        : (H,W) reference inverse depth (uint8 or float)
    fg_mask  : (H,W) bool, reference foreground mask (nearer than the local background).
               None (default) -> derived per component from the depth of the background
               that the component's disocclusion reveals (`bg_depth`, measured on the
               BG-classified boundary pixels): a pixel is foreground for that
               disocclusion iff P > bg_depth + `bg_margin`.  A supplied mask is used as an
               additional constraint (intersection).
    holes    : dict from `inverse_project_edges` (components carrying `ref_rows`) or a
               plain list of such component dicts.
    cfg      : RunConfig (uses `dilate_rm`)

    Returns
    -------
    dict(removed_mask (bool), removed_label (int32), components=[{id, case, bg_side,
         intervals, area, fallback, footprint_iou}], stats=..., fg_mask=..., fg_eff=...,
         classify_side=fn)
    """
    P = np.asarray(P, dtype=np.float32)
    H, W = P.shape
    comps = holes["components"] if isinstance(holes, dict) else holes
    fp_label = holes.get("footprint_label") if isinstance(holes, dict) else None
    if fg_mask is not None:
        fg_mask = _bool(fg_mask)
        if fg_mask.shape != P.shape:
            raise ValueError(f"fg_mask {fg_mask.shape} vs depth {P.shape}")

    dil = max(0, int(getattr(cfg, "dilate_rm", 2)))
    kernel = np.ones((2 * dil + 1, 2 * dil + 1), np.uint8) if dil else None

    removed = np.zeros((H, W), bool)
    removed_label = np.zeros((H, W), np.int32)
    fg_eff = np.zeros((H, W), bool)
    per_comp = []
    skipped_small = []
    n_fallback = n_empty = 0
    case_hist = Counter()

    for c in comps:
        i = int(c["id"])
        s = int(c["bg_side"])
        rows = c.get("ref_rows") or {}
        bg_depth = float(c.get("bg_depth", np.nan))
        # Guard: a disocclusion smaller than min_removal_area px (a handful of pixels at a
        # moving silhouette) has an inverse projection that is numerically unreliable: the
        # footprint can land on an unrelated surface and any band built from it becomes a
        # garbage strip far from the hole it was meant to fill (measured on cam6->cam7 f000:
        # a 55 px component with footprint coverage 0.00 whose removal had IoU 0.02).  Such
        # components are skipped here and left to the stage-5/6 postprocessing fallback; the
        # skip is counted, reported, and reflected in the coverage figure below.
        area_c = int(c.get("area", 0))
        if area_c < min_removal_area:
            skipped_small.append((i, area_c))
            continue
        if fg_mask is None:
            if not np.isfinite(bg_depth):
                mask_c = np.zeros((H, W), bool)
            else:
                mask_c = P > (bg_depth + float(bg_margin))
        else:
            mask_c = fg_mask.copy()
            if np.isfinite(bg_depth):
                mask_c &= P > (bg_depth + float(bg_margin))

        # eligible reference foreground around this component (for the masks / panels)
        if np.isfinite(bg_depth):
            nrow = max(1, int(getattr(cfg, "dilate_rm", 2)) + 2)
            for v, (lo, hi) in rows.items():
                if not (0 <= v < H):
                    continue
                a = max(0, int(lo) - 4)
                b = min(W, int(hi) + 5)
                v0, v1 = max(0, v - nrow), min(H, v + nrow + 1)
                fg_eff[v0:v1, a:b] |= P[v0:v1, a:b] > (bg_depth + float(bg_margin))

        case, nv, nr = removal_case(c)
        rm = np.zeros((H, W), bool)
        for v, (lo, hi) in rows.items():
            if v < 0 or v >= H:
                continue
            lo, hi = int(lo), int(hi)
            x0, x1 = lo, hi            # the band points from the background side inward
            if case == "B":
                # paper case 2: also remove every foreground pixel between the two
                # reference edge positions of the row
                o = (c.get("ref_opp") or {}).get(v)
                if o is not None and 0 <= int(o) < W:
                    x0, x1 = min(x0, int(o)), max(x1, int(o))
            if full_obj_in_case_b and case == "B":
                # stronger reading: the whole connected foreground region of the row
                fgr = mask_c[v]
                a, b = int(x0), int(x1)
                while a > 0 and fgr[a - 1]:
                    a -= 1
                while b < W - 1 and fgr[b + 1]:
                    b += 1
                x0, x1 = a, b
            x0 = max(0, x0)
            x1 = min(W - 1, x1)
            if x1 < x0:
                continue
            rm[v, x0:x1 + 1] |= mask_c[v, x0:x1 + 1]

        used_fallback = ""
        if not rm.any():
            # fallback 1 -- ignore the global mask, use the component's own background
            n_fallback += 1
            used_fallback = "local-bg-depth"
            if np.isfinite(bg_depth):
                m2 = P > (bg_depth + 2.0)
                for v, (lo, hi) in rows.items():
                    if 0 <= v < H:
                        lo2, hi2 = max(0, int(lo)), min(W - 1, int(hi))
                        rm[v, lo2:hi2 + 1] |= m2[v, lo2:hi2 + 1]
        if not rm.any():
            # fallback 2 -- the nearest foreground run inside the footprint rows
            n_fallback += 1
            used_fallback = "max-depth-run"
            for v, (lo, hi) in rows.items():
                if not (0 <= v < H):
                    continue
                lo2, hi2 = max(0, int(lo)), min(W - 1, int(hi))
                seg = P[v, lo2:hi2 + 1]
                if seg.size and seg.max() > 0:
                    rm[v, lo2:hi2 + 1] |= (seg >= 0.75 * float(seg.max()))

        if dil and rm.any():
            rm = cv2.dilate(rm.astype(np.uint8), kernel) > 0
        rm &= mask_c
        # intervals are read back from the FINAL mask (after dilation + FG intersection)
        intervals = []
        for v in rows:
            if 0 <= v < H:
                xs = np.nonzero(rm[v])[0]
                if xs.size:
                    intervals.append((int(v), int(xs.min()), int(xs.max())))
        if not rm.any():
            n_empty += 1
        removed |= rm
        removed_label[rm] = i

        iou = float("nan")
        if fp_label is not None:
            fp = fp_label == i
            u = int((rm & fp).sum())
            o = int((rm | fp).sum())
            iou = (u / float(o)) if o else float("nan")
        case_hist[case if used_fallback == "" else case + "*"] += 1
        per_comp.append(dict(id=i, case=case, bg_side=s, intervals=intervals,
                             area=int(rm.sum()), fallback=used_fallback,
                             footprint_iou=iou, bg_depth=bg_depth,
                             n_case_votes=nv, n_case_rows=nr,
                             bbox=c.get("bbox"), fp_bbox=c.get("fp_bbox")))

    def classify_side(x, y):
        if x < 0 or x >= W or y < 0 or y >= H:
            return None
        if fg_eff.any():
            return 1 if fg_eff[y, x] else 0
        if fg_mask is not None:
            return 1 if fg_mask[y, x] else 0
        return None

    stats = dict(n_components=len(per_comp), n_empty=n_empty, n_fallback=n_fallback,
                 removed_px=int(removed.sum()), case_hist=dict(case_hist),
                 dilate_rm=dil, fg_eff_px=int(fg_eff.sum()),
                 n_skipped_small=len(skipped_small), skipped_small=skipped_small)
    return dict(removed_mask=removed, removed_label=removed_label, components=per_comp,
                stats=stats, fg_mask=fg_mask, fg_eff=fg_eff, classify_side=classify_side)


# --------------------------------------------------------------------------- #
# 4. depth prediction for the removed region  (paper III-D, eq. 4-5)
# --------------------------------------------------------------------------- #
def _runs(row_bool):
    """[(x0, x1)] inclusive runs of True in a 1-D boolean row."""
    idx = np.nonzero(row_bool)[0]
    if idx.size == 0:
        return []
    brk = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], brk + 1))
    ends = np.concatenate((brk, [idx.size - 1]))
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


def _nearest_valid_on_row(depth, row, x0, x1):
    """Nearest depth>0 pixel outside [x0,x1] on the row (left first, then right)."""
    W = depth.shape[1]
    left = depth[row, :x0]
    li = np.nonzero(left > 0)[0]
    right = depth[row, x1 + 1:W]
    ri = np.nonzero(right > 0)[0]
    dl = float(left[li[-1]]) if li.size else None
    dr = float(right[ri[0]]) if ri.size else None
    return dl, dr


def predict_removed_depth(P, removed_mask, cams, src, dst, classify_side_fn=None,
                          fg_mask=None, info=None, bg_level=None,
                          enforce_bg_level=True, bg_tol=15.0, comp_label=None,
                          comp_bg_level=None):
    """Predict the depth of the removed region, paper eq. (4)-(5).

    Per reference row, per removed run [u_l, u_r]:
      both edge pixels BG        -> linear interpolation between their depth values
      left BG, right FG          -> constant d(u_l)
      left FG, right BG          -> constant d(u_r)
      no usable edge on a side   -> nearest valid depth on that row

    `classify_side_fn(x, y)` must return 1 (foreground) / 0 (background) / None for the
    two pixels just outside the run; it is expected to use the SAME local rule as the
    removal (nearer than the background that the disocclusion reveals), never a global
    window mask.  A run whose both endpoints are foreground (a fragment of the object
    separated from the background-side endpoint by a small background sliver) belongs to
    the background layer that the disocclusion exposes, so `bg_level` (per pixel local
    background level, NaN where unknown) is used; the "nearest valid on the row" rule then
    only fires where a side is genuinely undefined (out of image or P == 0).

    `enforce_bg_level` (default on) adds the paper's own stated assumption as a final
    per-component correction: "the removed region and its surrounding background content
    belong to the same physical surface, so they should have similar depth values".  If a
    component's predicted median still disagrees with its locally measured revealed
    background level by more than `bg_tol`, the component is set to that background level.
    Without it, a mis-classified run endpoint can silently pull a depth from a NEARER
    layer on some frames, which makes the reconstruction land in the wrong place.

    Only removed pixels are written.  Returns (P_pred uint8, info).
    """
    P = np.asarray(P, dtype=np.float64)
    H, W = P.shape
    rem = _bool(removed_mask)
    if rem.shape != P.shape:
        raise ValueError(f"removed_mask {rem.shape} vs depth {P.shape}")
    out = P.copy()

    if classify_side_fn is None and fg_mask is not None:
        fgm = _bool(fg_mask)

        def classify_side_fn(x, y, _m=fgm):
            if 0 <= x < _m.shape[1] and 0 <= y < _m.shape[0]:
                return 1 if _m[y, x] else 0
            return None

    cases = Counter()
    row_cases = {}
    n_runs = 0
    n_written = 0

    for v in range(H):
        if not rem[v].any():
            continue
        for (x0, x1) in _runs(rem[v]):
            n_runs += 1
            lx, rx = x0 - 1, x1 + 1
            dl_near, dr_near = _nearest_valid_on_row(P, v, x0, x1)
            # an edge pixel is usable only if it exists AND has a defined depth
            cl = classify_side_fn(lx, v) if (classify_side_fn is not None
                                             and lx >= 0 and P[v, lx] > 0) else None
            cr = classify_side_fn(rx, v) if (classify_side_fn is not None
                                             and rx < W and P[v, rx] > 0) else None
            dl = float(P[v, lx]) if (lx >= 0 and P[v, lx] > 0) else dl_near
            dr = float(P[v, rx]) if (rx < W and P[v, rx] > 0) else dr_near
            bg = float(bg_level[v, x0:x1 + 1].mean()) if bg_level is not None else np.nan

            u = np.arange(x0, x1 + 1, dtype=np.float64)
            if cl == 0 and cr == 0 and x1 > x0:
                s = (dr - dl) / float(x1 - x0)
                vals = dl + s * (u - x0)
                case = "BG-BG"
            elif cl == 0 and dl is not None:
                vals = np.full(u.size, dl)
                case = "BG-FG"
            elif cr == 0 and dr is not None:
                vals = np.full(u.size, dr)
                case = "FG-BG"
            elif np.isfinite(bg) and (cl == 1 or cr == 1):
                # both endpoints foreground: the run is a foreground fragment whose
                # background continuation is what the hole reveals -> background level
                vals = np.full(u.size, bg)
                case = "FG-FG(bg-level)"
            elif dl is not None and dr is not None:
                near_l = (u - lx) <= (rx - u)
                vals = np.where(near_l, dl, dr)
                case = "nearest"
            elif dl is not None:
                vals = np.full(u.size, dl)
                case = "nearest-left"
            elif dr is not None:
                vals = np.full(u.size, dr)
                case = "nearest-right"
            else:
                vals = np.zeros(u.size)
                case = "zero"
            out[v, x0:x1 + 1] = np.clip(vals, 0, 255)
            n_written += (x1 - x0 + 1)
            cases[case] += 1
            row_cases[int(v)] = case

    changed = out != P
    n_outside = int((changed & (~rem)).sum())
    n_unfilled = int((rem & (out == P) & (P == 0)).sum())   # removed px still 0
    inf = dict(info or {})
    inf.update(n_runs=n_runs, n_written_px=n_written, case_hist=dict(cases),
               row_cases=row_cases, n_changed_outside=int(n_outside),
               n_removed=int(rem.sum()), n_removed_left_zero=n_unfilled,
               n_changed=int(changed.sum()))

    # ---- final per-component consistency pass (see docstring) ---------------- #
    if enforce_bg_level and rem.any():
        # `comp_bg_level` maps a removal component id -> the background depth that THIS
        # disocclusion reveals, measured from its own BG-classified edge pixels.  Using the
        # per-component value (rather than only the rasterised level map, which can be
        # ambiguous where two disocclusions share a row) makes the correction unambiguous:
        # a component's prediction must sit at ITS OWN revealed background level.
        use_comp = comp_label is not None and comp_bg_level
        if use_comp or bg_level is not None:
            fixed = []
            if use_comp:
                # iterate the REMOVAL-pass components so that comp_label ids and
                # comp_bg_level keys are the same labelling (a labelling of the final
                # boolean mask merges bands that touch, which would look up the wrong level)
                for cid, lvl in comp_bg_level.items():
                    m = comp_label == cid
                    if not m.any() or not np.isfinite(lvl):
                        continue
                    med_pred = float(np.median(out[m]))
                    # Conservative direction: only ever move the prediction TOWARDS the
                    # background (farther == smaller P).  A component's measured "revealed
                    # background level" can itself be too near when its background-side edge
                    # sample was misclassified as foreground, and trusting it blindly would
                    # then push the whole component onto the wrong (nearer) layer -- which is
                    # exactly the failure this pass exists to prevent.
                    if med_pred > float(lvl) + bg_tol:
                        out[m] = float(lvl)
                        fixed.append((int(cid), med_pred, float(lvl)))
            else:
                nlab, cl = cv2.connectedComponents(rem.astype(np.uint8), connectivity=8)
                for cid in np.unique(cl[rem]):
                    if cid == 0:
                        continue
                    m = cl == cid
                    lv = bg_level[m]
                    lv = lv[np.isfinite(lv)]
                    if lv.size == 0:
                        continue
                    lvl = float(np.median(lv))
                    med_pred = float(np.median(out[m]))
                    if med_pred > lvl + bg_tol:
                        out[m] = lvl
                        fixed.append((int(cid), med_pred, lvl))
            inf["bg_level_corrections"] = fixed
        # pointwise clamp: the paper uses the (predicted) depth "to prevent foreground
        # penetration".  A run whose endpoint classification is wrong can still leave a few
        # pixels at a nearer layer than the surface they belong to, so no predicted pixel is
        # allowed to sit more than `bg_tol/2` above its own local revealed background level.
        # This is a clamp, never a lift: it can only push the prediction towards the
        # background, which is the direction the paper requires.
        ok = np.isfinite(bg_level) & rem if bg_level is not None else np.zeros_like(rem)
        if ok.any():
            cap = bg_level + bg_tol * 0.5
            over = ok & (out > cap)
            n_clamped = int(over.sum())
            out[over] = cap[over]
            inf["bg_level_clamped_px"] = n_clamped
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), inf


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def describe(edges, footprints=None, removal=None, pred_info=None):
    """Human readable multi-line summary used by the step scripts / report."""
    s = edges["stats"]
    lines = [f"disocclusions           : {s['n_components']}",
             f"disocclusion edge px    : {s['n_boundary_pixels']}",
             f"undetermined edge px    : {s['n_undetermined_pixels']}",
             f"laplacian vs depth agree: {s['agree_n']}/{s['agree_n_able']} = "
             f"{s['agree_rate']:.3f}" if s["agree_n_able"] else
             "laplacian vs depth agree: n/a",
             f"mean depth of F edges   : {s['mean_depth_F']:.2f}",
             f"mean depth of BG edges  : {s['mean_depth_BG']:.2f}",
             f"background-side counts  : {s['bg_side_counts']}"]
    if footprints is not None:
        lines.append(f"reference footprint px  : {footprints['stats']['n_footprint_px']} "
                     f"(no footprint: {footprints['stats']['n_no_footprint']})")
    if removal is not None:
        lines.append(f"removed foreground px   : {removal['stats']['removed_px']} "
                     f"(cases {removal['stats']['case_hist']}, "
                     f"fallbacks {removal['stats']['n_fallback']}, "
                     f"empty {removal['stats']['n_empty']})")
    if pred_info is not None:
        lines.append(f"depth prediction        : {pred_info['n_runs']} runs, "
                     f"cases {pred_info['case_hist']}")
    return "\n".join(lines)
