"""Forward 3D warping (DIBR) with sub-pixel splatting + Z-buffer, plus the derived maps
(backward index, crack detection, OOFA / disocclusion separation).

Conventions
-----------
* colour images are RGB, float or uint8, shape (H, W, 3); a single channel is allowed.
* `z` is the ordering value with **LARGER = NEARER** (i.e. the inverse depth P, or z_inv).
* A destination pixel with no sample is a hole (`hole_mask = True`).
"""
import cv2
import numpy as np

from . import calib


# --------------------------------------------------------------------------- #
# splat kernels
# --------------------------------------------------------------------------- #
def _splat_2d(dx, dy):
    h, w = dx.shape
    y, x = np.mgrid[0:h, 0:w]
    tx = x + dx
    ty = y + dy
    x0 = np.floor(tx).astype(np.int32)
    y0 = np.floor(ty).astype(np.int32)
    wx = (tx - x0).astype(np.float32)
    wy = (ty - y0).astype(np.float32)
    xi = x.astype(np.int32)
    yi = y.astype(np.int32)
    out = []
    for ox, oy, ww in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                       (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        out.append((x0 + ox, y0 + oy, ww, xi, yi))
    return out


def _splat_single(dx, dy, mode="floor"):
    h, w = dx.shape
    y, x = np.mgrid[0:h, 0:w]
    f = np.floor if mode == "floor" else np.round
    tx = f(x + dx).astype(np.int32)
    ty = f(y + (0.0 if dy is None else dy)).astype(np.int32)
    return [(tx, ty, np.ones((h, w), np.float32), x.astype(np.int32), y.astype(np.int32))]


# --------------------------------------------------------------------------- #
# main forward warp
# --------------------------------------------------------------------------- #
def forward_warp(color, dx, dy=None, z=None, hole_depth=-1.0, rule="zbuf",
                 splat="sub", return_index=False):
    """Splat every source pixel to `src + (dx, dy)`.

    Parameters
    ----------
    color : (H, W, C) or (H, W)
    dx, dy : (H, W) float  signed displacement, target = source + d
    z : (H, W) float  ordering value, LARGER = NEARER (inverse depth).  Used by the
        Z-buffer.  Defaults to 1 everywhere (pure ordered splatting).
    rule : "zbuf" (nearest layer wins -> correct occlusion) or "avg" (accumulate all)
    splat : "sub" (2x2 or 1x2 sub-pixel) or "floor"/"round" (one integer target pixel,
        the classical DIBR splat that produces 1-2 px cracks)

    Returns
    -------
    warped_color (H,W,C) float32, warped_z (H,W) float32 (hole_depth where empty),
    hole_mask (H,W) bool, weight (H,W) float32, index (H,W) int32 (only if requested;
        flat source index of the winning sample, -1 where empty)
    """
    color = np.asarray(color, dtype=np.float32)
    dx = np.asarray(dx, dtype=np.float32)
    h, w = dx.shape
    if color.shape[:2] != (h, w):
        raise ValueError(f"shape mismatch: color {color.shape} vs displacement {(h, w)}")
    c = color.shape[2] if color.ndim == 3 else 1
    src = color.reshape(h, w, c)
    if z is None:
        z = np.ones((h, w), np.float32)
    else:
        z = np.asarray(z, dtype=np.float32)

    if splat == "sub":
        contribs = _splat_1d(dx) if dy is None else _splat_2d(dx, np.asarray(dy, np.float32))
    else:
        contribs = _splat_single(dx, dy, splat)

    packed = []
    for tx, ty, wt, sx, sy in contribs:
        sel = (wt > 0) & (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        if sel.any():
            packed.append((tx[sel], ty[sel], wt[sel], z[sel], sx[sel], sy[sel]))

    if rule == "zbuf":
        zbuf = np.full((h, w), -np.inf, np.float32)
        for tx, ty, _wt, zz, _sx, _sy in packed:
            np.maximum.at(zbuf, (ty, tx), zz)
        used = []
        for tx, ty, wt, zz, sx, sy in packed:
            keep = zz >= zbuf[ty, tx]
            used.append((tx[keep], ty[keep], wt[keep], zz[keep], sx[keep], sy[keep]))
    else:
        used = packed

    wacc = np.zeros((h, w), np.float32)
    cacc = np.zeros((h, w, c), np.float32)
    zacc = np.zeros((h, w), np.float32)
    for tx, ty, wt, zz, sx, sy in used:
        if tx.size == 0:
            continue
        np.add.at(cacc, (ty, tx), src[sy, sx] * wt[:, None])
        np.add.at(wacc, (ty, tx), wt)
        np.add.at(zacc, (ty, tx), zz * wt)

    valid = wacc > 0
    warped_color = np.zeros_like(cacc)
    warped_color[valid] = cacc[valid] / wacc[valid][:, None]
    warped_z = np.full((h, w), hole_depth, np.float32)
    warped_z[valid] = zacc[valid] / wacc[valid]
    if c == 1:
        warped_color = warped_color[:, :, 0]
    hole_mask = ~valid

    if not return_index:
        return warped_color, warped_z, hole_mask, wacc

    idx = np.full((h, w), -1, np.int32)
    for tx, ty, _wt, zz, sx, sy in used:
        if tx.size == 0:
            continue
        # keep the same winner rule as the colour accumulation
        sel = zz >= zbuf[ty, tx]
        idx[ty[sel], tx[sel]] = (sy[sel].astype(np.int32) * w + sx[sel].astype(np.int32))
    return warped_color, warped_z, hole_mask, wacc, idx


def _splat_1d(dx):
    h, w = dx.shape
    y, x = np.mgrid[0:h, 0:w]
    t = x + dx
    x0 = np.floor(t).astype(np.int32)
    wx = (t - x0).astype(np.float32)
    yi = y.astype(np.int32)
    xi = x.astype(np.int32)
    return [(x0, yi, 1.0 - wx, xi, yi), (x0 + 1, yi, wx, xi, yi)]


# --------------------------------------------------------------------------- #
# coverage / boundary derived maps
# --------------------------------------------------------------------------- #
def coverage_mask(shape, dx, dy=None, dilate=1):
    """Concave hull of the warped point cloud: pixels inside the warped footprint.

    Implemented as a binary closing of the "non-empty" mask, iterated until stable,
    so that the interior of a disocclusion (bounded on both sides by warped content)
    counts as covered while OOFA (open to the boundary) does not.
    """
    h, w = shape
    def _splat_ones(dx_, dy_):
        dx_ = np.asarray(dx_, np.float32)
        if dy_ is None:
            contribs = _splat_1d(dx_)
        else:
            contribs = _splat_2d(dx_, np.asarray(dy_, np.float32))
        acc = np.zeros((h, w), np.float32)
        for tx, ty, wt, _sx, _sy in contribs:
            sel = (wt > 0) & (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
            np.add.at(acc, (ty[sel], tx[sel]), wt[sel])
        return acc > 0
    empty = ~_splat_ones(dx, dy)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    cur = (~empty).astype(np.uint8)
    for _ in range(64):
        nxt = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, k)
        if not (nxt != cur).any():
            break
        cur = nxt
    return cur.astype(bool)


def interior_gap(mask, dilate=1):
    """Split the hole mask into holes enclosed by warped content and holes open to the border.

    Returns (enclosed, oofa) where both are subsets of `mask`:
      * `oofa`   : hole components that touch the image border -- the "out of field area",
                   i.e. the part of the virtual view the reference camera never captured.
      * `enclosed`: every other hole pixel (disocclusions and interior cracks).

    A hole component that merely *reaches* the border counts as OOFA, which is what the
    paper's postprocessing leaves alone.
    """
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    enclosed = np.zeros((h, w), bool)
    oofa = np.zeros((h, w), bool)
    if not mask.any():
        return enclosed, oofa
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    border_labels = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) | \
                    set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    border_labels.discard(0)
    # also treat a component whose bounding box spans the whole frame as border-open
    for i in range(1, n):
        x, y, bw, bh = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        if bw >= w or bh >= h:
            border_labels.add(i)
    if border_labels:
        oofa = np.isin(lab, list(border_labels))
    enclosed = mask & ~oofa
    return enclosed, oofa


def crack_mask(hole_mask, max_width=2):
    """Holes whose local thickness <= max_width (the 1-2 px 'cracks' of integer DIBR)."""
    hole_u8 = np.asarray(hole_mask, np.uint8)
    if not hole_u8.any():
        return np.zeros_like(hole_mask, bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole_u8, connectivity=8)
    dist = cv2.distanceTransform(hole_u8, cv2.DIST_L2, 3)
    out = np.zeros_like(hole_mask, bool)
    for i in range(1, n):
        m = lab == i
        if dist[m].max() <= max_width:
            out |= m
    return out


def nearest_valid_index(valid):
    """For every pixel, the flat index of the nearest `valid` pixel (itself if valid)."""
    from scipy import ndimage
    valid = np.asarray(valid, bool)
    if not valid.any():
        raise ValueError("no valid pixel to propagate from")
    _dist, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return iy.astype(np.int32), ix.astype(np.int32)


def fill_cracks(color, crack):
    """Fill thin hole regions with the nearest valid colour ('surrounding valid pixels').

    Only writes pixels inside `crack`; never touches real disocclusions.
    """
    crack = np.asarray(crack, bool)
    out = np.array(color, copy=True)
    if not crack.any():
        return out
    iy, ix = nearest_valid_index(~crack)
    yy, xx = np.nonzero(crack)
    out[yy, xx] = color[iy[yy, xx], ix[yy, xx]]
    return out


# --------------------------------------------------------------------------- #
# backward index (\"which source pixel landed here\")
# --------------------------------------------------------------------------- #
def backward_index(shape, dx, dy=None, z=None):
    """Return a flat source index per destination pixel (-1 where empty)."""
    h, w = shape
    _, _, _, _, idx = forward_warp(np.zeros((h, w), np.float32), dx, dy, z=z,
                                   return_index=True)
    return idx


# --------------------------------------------------------------------------- #
# high level: warp a reference view to the virtual camera
# --------------------------------------------------------------------------- #
def warp_view(cams, src, dst, color, P):
    """Warp `color` + depth `P` (uint8) from cam `src` to cam `dst`.

    Returns dict with warped colour/depth, hole mask, warped inverse-depth (for
    ordering), backward index and the warped Laplacian of the depth image.
    """
    P = np.asarray(P, np.uint8)
    z = calib.depth_from_P(P)
    z_inv = 1.0 / z                            # larger = nearer, for the Z-buffer
    dx, dy, u_t, v_t, oob = calib.displacement_field(cams, src, dst, P)
    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)
    oob = ~oob

    color_f = color.astype(np.float32)
    wcolor, wz, hole, _w, idx = forward_warp(color_f, dx, dy, z=z_inv, rule="zbuf",
                                            splat="sub", return_index=True)
    wdepth, wz2, hole_d, _ = forward_warp(P.astype(np.float32), dx, dy, z=z_inv,
                                         rule="zbuf", splat="sub", hole_depth=-1.0)
    # warped Laplacian of the (preprocessed) depth image, eq. (3)
    lap = cv2.Laplacian(P.astype(np.float32), cv2.CV_32F, ksize=3)
    wlap, _, hole_l, _ = forward_warp(lap, dx, dy, z=z_inv, rule="zbuf", splat="sub",
                                      hole_depth=0.0)
    return dict(
        warped_color=np.clip(wcolor, 0, 255).astype(np.uint8),
        warped_depth=np.where(hole_d, -1.0, wdepth).astype(np.float32),
        warped_invdepth=wz.astype(np.float32),
        warped_lap=wlap.astype(np.float32),
        hole=hole,
        backward=idx,
        dx=dx, dy=dy, u_t=u_t.astype(np.float32), v_t=v_t.astype(np.float32),
        src_oob=oob,                       # source pixel projected outside the destination
    )
