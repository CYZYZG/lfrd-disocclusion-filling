"""Modified Criminisi exemplar-based inpainting -- paper III-D, eq. (6)-(10).

This module fills the region that stage B removed from the *reference* view (the
local foreground strip that occluded the disocclusion).  The predicted occlusion layer
produced here is later warped into the virtual view (stage 6 / ``lfrd.render``), which is
the core idea of the paper: disocclusions are never inpainted in the virtual image.

Algorithm (paper, section III-D)
--------------------------------
priority, eq. (6)::

    P(p) = [ C(p) . D(p) . Z(p) ] . B(p)

    C(p)  Criminisi confidence: sum of C over the patch valid part / |Psi_p|
    D(p)  Criminisi data term:  |nabla(I)_p^perp . n_p| / alpha
    Z(p)  eq. (7) depth term:   (d_max - mean_{q in Psi_p ^ source} d(q)) / (d_max - d_min)
    B(p)  eq. (8) background term: 0 if p is foreground, 1 if p is background

matching, eq. (9)-(10)::

    Psi_q_hat = argmin_{Psi_q in S'} SSD_color(Psi_p_hat, Psi_q)  s.t.  DD <= depth_tol
    DD = |Z_q - Z_p| / Z_p

The foreground / background classification B(p) follows the paper's paragraph after
eq. (8): the Laplacian operator is applied to the *predicted* depth image and the sign
of (nabla d) decides the class -- (nabla d) < 0 is foreground (the nearer layer, larger
inverse depth P).  Because the paper's delta-Omega is the *edge band* of the removed
region (which contains both the removed pixels and the valid pixels they border), an edge
pixel classified foreground labels the hole pixels that touch it: the front pixel whose
patch abuts a foreground edge pixel is "the relevant foreground pixel" whose priority is
set to 0 (``fg_dilate=1`` dilates the foreground edge over exactly those pixels).
``fg_mask=None`` derives it this way from ``depth``; callers may pass their own mask.

Implementation notes (and why it is fast)
-----------------------------------------
* Every iteration evaluates the whole local search window (160x120 candidates, paper
  value) *vectorised*.  The masked SSD of eq. (9) is rewritten as three correlations::

      SSD(q) = A - 2 * sum_{o in M} <I(p+o), I(q+o)>  +  sum_{o in M} ||I(q+o)||^2
             = A - 2 * corr(I, M*I_p)                +  corr(I^2, M)

  computed with ``cv2.filter2D`` (a correlation in OpenCV) on the local region.  The
  depth patch mean Z_q of eq. (10) is the same correlation applied to the depth image,
  so no integral image bookkeeping is needed and nothing is done in a Python double loop.
  Measured cost: ~0.35 ms per iteration for a 160x120 window on 1024x768 (single thread).
* Priorities are maintained lazily: filling one patch can only change the front within
  ``2*patch_size`` pixels of it, so only that window is refreshed and only changed
  priorities are pushed on the heap.
* The whole input is padded by ``patch_size//2`` so that no patch bookkeeping needs
  clipping; the padding is marked invalid, which also rejects candidate patches that
  would leave the image.

Conventions
-----------
* ``image`` RGB (or gray) uint8/float; ``hole_mask`` True = to be filled;
  ``depth`` uint8/float **inverse** depth, LARGER = nearer = foreground, 0 = undefined.
* Only pixels inside ``hole_mask`` are ever written.

Documented deviations from the literal text (all switchable)
------------------------------------------------------------
1. ``layer_ref="hole"`` (default): eq. (9)-(10) compare Z_q with the mean *predicted*
   depth of the hole part of the target patch instead of the mean over its valid part.
   The paper's own argument ("it is necessary to select candidate patches located in the
   same depth layer as the patch to be inpainted"; "the removed region belongs to the
   background and should be filled by background texture") requires the *background*
   layer, but the valid part of a patch that touches the surviving foreground person is
   dominated by that person's near depth, so the literal reading makes the DD test prefer
   foreground candidates.  Measured on the synthetic fixture: 46% of filled pixels took
   their colour from the foreground region with ``layer_ref="valid"`` vs ~0% with
   ``"hole"``.  ``layer_ref="valid"`` restores the literal reading.
2. ``layer_guard=True`` (default): an extra candidate filter -- no pixel of the candidate
   patch may be nearer than ``1 + 3*depth_tol`` times the target layer depth.  This makes
   "same depth layer" hold for *every* copied pixel, not only on average.  A patch whose
   mean passes DD but which straddles the person is rejected.
3. The priority Z(p) of eq. (7) keeps the literal definition (mean over the patch source
   part), which is what makes the foreground side of the front *less* urgent.
4. ``branch_hist`` reports which rung of the matching fallback ladder was used
   (0 = strict DD + layer guard, 1/2/3 = DD relaxed x2/x4/x8 with the guard, 4 = DD
   dropped (guard kept), 5 = guard dropped, 6 = DD and guard dropped, 7 = global search,
   8 = nearest-valid copy).  The guard is relaxed *last* because it is what keeps
   foreground texture out of the copies.  The paper's eq. (4) cases belong to the depth
   prediction of stage B, not to this module.
"""
from __future__ import annotations

import heapq
import time

import cv2
import numpy as np

from . import warp as _warp

_CROSS3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def as_bool(mask):
    """uint8 (255 = on) or bool or any numeric mask -> bool array."""
    m = np.asarray(mask)
    if m.dtype == bool:
        return m
    return m > 0


def luminance(img):
    """RGB uint8/float -> float32 gray."""
    f = np.asarray(img, np.float32)
    if f.ndim == 2:
        return f
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def laplacian_fg(depth, dilate=1, valid=None):
    """Foreground classification of the removed-region edge (eq. 3 / eq. 8 paragraph).

    ``(nabla d) < 0`` -> foreground (nearer layer, larger P).  ``dilate`` grows the edge
    label so that the *hole* pixels abutting a foreground edge pixel are labelled too.
    Returns ``(fg, lap)`` with ``fg`` bool and ``lap`` the Laplacian.
    """
    d = np.asarray(depth, np.float32)
    lap = cv2.Laplacian(d, cv2.CV_32F, ksize=3)
    fg = lap < 0
    if valid is not None:
        v = as_bool(valid)
        lap = np.where(v, lap, 0.0)
        fg &= v
    if dilate and dilate > 0:
        k = np.ones((2 * int(dilate) + 1,) * 2, np.uint8)
        fg = cv2.dilate(fg.astype(np.uint8), k) > 0
    return fg, lap


def layer_fg_mask(depth, valid=None, min_gap=8):
    """Two-layer split of a depth map (evaluation helper, NOT part of the paper).

    Used by the step-5 checks to *measure* whether filled pixels were sampled from the
    foreground person.  Threshold = Otsu on the nonzero inverse-depth histogram (a
    fallback of ``median + min_gap`` is used when Otsu is degenerate).
    Returns ``(fg, thr)``.
    """
    d = np.asarray(depth, np.float32)
    v = (d > 0) if valid is None else (as_bool(valid) & (d > 0))
    if not v.any():
        return np.zeros_like(d, bool), 0.0
    vals = np.clip(d[v], 0, 255)
    hist = np.bincount(vals.astype(np.int32), minlength=256).astype(np.float64)
    tot = hist.sum()
    idx = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = tot - w0
    m0 = np.cumsum(hist * idx)
    m1 = m0[-1] - m0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, m0 / np.maximum(w0, 1), 0.0)
        mu1 = np.where(w1 > 0, m1 / np.maximum(w1, 1), 0.0)
        var = w0 * w1 * (mu0 - mu1) ** 2
    thr = float(np.argmax(var))
    med = float(np.median(vals))
    if not np.isfinite(thr) or thr <= med or thr > vals.max():
        thr = med + float(min_gap)
    return (d >= thr) & v, thr


def _nearest_valid_fill(arr, hole):
    """Replace ``hole`` pixels of ``arr`` by the nearest valid value (for gradients)."""
    out = np.array(arr, copy=True)
    h = as_bool(hole)
    if not h.any() or h.all():
        return out
    iy, ix = _warp.nearest_valid_index(~h)
    out[h] = arr[iy[h], ix[h]]
    return out


def _patch_ok(valid0, ps):
    """Candidate patch centres whose full ``ps x ps`` window is inside the source region."""
    r = ps // 2
    Hp, Wp = valid0.shape
    pad = np.zeros((Hp + 2 * r, Wp + 2 * r), np.uint8)
    pad[r:r + Hp, r:r + Wp] = valid0.astype(np.uint8)
    er = cv2.erode(pad, np.ones((ps, ps), np.uint8))
    return er[r:r + Hp, r:r + Wp] > 0


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def inpaint(image, hole_mask, depth, patch_size=9, search_w=160, search_h=120,
            alpha=255, depth_tol=0.2, use_bg_term=True, use_depth_term=True,
            use_depth_limit=True, local_search=True, fg_mask=None, max_iters=200000,
            seed=None, return_meta=False, fg_dilate=1, record_source=False,
            layer_ref="hole", layer_guard=True, layer_depth=None,
            verbose=False, progress_every=0, log=None,
            sizes=None, beta=35.0, beta_mode="mean", struct_pen=0.0,
            size_rule="best", size_bonus=0.15, template_exclude_fg=False, fg_tol=8.0):
    """Modified Criminisi inpainting of ``hole_mask`` in ``image`` (paper III-D).

    Parameters
    ----------
    image : (H,W,3) or (H,W) array, RGB (or gray), uint8 or float
    hole_mask : (H,W) True = pixel to be filled (only these are ever written)
    depth : (H,W) inverse depth, LARGER = nearer = foreground, 0 = undefined
    patch_size, search_w, search_h, alpha, depth_tol : paper IV-A values
    use_bg_term : eq. (8) B(p) (if False, B == 1 == the postprocessing mode)
    use_depth_term : eq. (7) Z(p) in the priority (if False, Z == 1)
    use_depth_limit : eq. (9)-(10) DD <= depth_tol constraint on candidates
    local_search : True -> 160x120 window around p (paper "global -> local"), clamped to
        the image; False -> the whole image is searched
    fg_mask : bool, True where foreground; None -> derive from the Laplacian of ``depth``
    max_iters : hard iteration cap
    seed : unused (the algorithm is deterministic); accepted for API compatibility
    return_meta : False -> return the filled image, True -> return the meta dict
    fg_dilate : dilation of the Laplacian foreground edge before B(p) is applied
    record_source : also return a map of the source pixel copied into every filled pixel
    layer_ref : "hole" (default) -> Z_p of eq. (9) is the mean *predicted* depth of the
        hole part of the patch, i.e. the background layer the removed region belongs to;
        "valid" -> the literal mean over the patch valid part (see the deviation note in
        the module docstring)
    layer_depth : optional (H,W) depth map used **only** as the eq. (9) layer reference for
        the hole part (instead of ``depth``).  Needed when the upstream depth prediction
        left the removed region at the foreground depth; see step5_inpaint.py.
    layer_guard : also require that no pixel of the candidate patch is more than
        ``1 + 3*depth_tol`` times the target layer depth (rejects patches that touch the
        foreground person even when their mean depth happens to pass DD)
    template_exclude_fg : drop template pixels that are clearly foreground (depth above the
        separator between the dense background level and the top-decile foreground level)
        before computing the SSD.  The B(p) priority already stops the fill from *starting*
        on foreground, but once the front has advanced the query patch still straddles the
        silhouette; those few foreground samples then dominate the SSD and drive the matcher
        to reproduce the dark object edge inside the hole (a halo along the seam).  Such
        pixels are excluded from the template only, never from the candidate sources.
    sizes : optional iterable of odd patch sizes, tested LARGEST FIRST for every iteration.
        The first size whose best admissible match has a per-pixel-per-channel mean SSD
        ``beta`` or less is accepted; if none is good enough the smallest size is used.
        This is the "adaptive patch size" of the sibling DIBR paper: a large patch gives
        coherent structure where the texture is smooth, while the cascade falls back to a
        tiny patch where no large match exists (which is what stops the blurry smearing of
        a single fixed 9x9 patch).  ``None`` (default) = the paper's single ``patch_size``.
    beta, beta_mode : acceptance threshold for ``sizes``.  "mean" (default) compares
        SSD / (C * n_valid), i.e. a mean squared per-channel error in gray levels, which is
        the only scale on which a fixed threshold is meaningful on 8-bit data; "sum" is the
        literal SSD.
    size_rule : "best" (default) -> every size is evaluated and the lowest mean-SSD/channel
        wins, with a small bonus for larger patches (see size_bonus); "cascade" -> the
        sibling paper's rule, accept the first size whose cost is <= beta, largest first.
    size_bonus : preference for larger patches under size_rule="best"; the score is
        cost * (1 - size_bonus * k/max(sizes)), so 0.15 makes a 9x9 with a cost up to ~15%
        higher win over a 3x3 patch.  Measured: the sibling's early-accept rule makes the
        3x3 pre-empt the 9x9 far too often on this data (cam5->cam4 f000 disocclusion PSNR
        19.21 dB vs 19.57 dB for the single 9x9), while "best" keeps the structure and still
        fixes the cases where no 9x9 exists.
    struct_pen : cross-row structural penalty.  A vertical offset of ``dy`` pixels is
        charged ``struct_pen * w * dy^2 * (C * n_valid)``, where ``w`` is the strength of
        HORIZONTAL structure around the hole (mean |vertical gradient| over a 15x15 window,
        clamped to 3).  With vertical structure present (barres, rails, fences) borrowing a
        source patch from another row is expensive, so horizontal structures stay aligned;
        in a vertically homogeneous background (curtain, floor) ``w ~ 0`` and horizontal
        borrowing stays free.  0 (default) reproduces the paper's plain SSD.
    progress_every : print a progress line every N iterations (0 = never)
    log : callable(str) used for progress lines (default print)

    Returns
    -------
    filled image (same shape/dtype class as the input, uint8) if ``return_meta`` is False,
    else a dict with the keys
    ``filled, filled_depth, n_iters, seconds, priority_at_first, branch_hist,
    candidates_rejected_no_depth, candidates_rejected_none`` plus useful extras.
    """
    t0 = time.perf_counter()
    src_img = np.asarray(image)
    if src_img.ndim == 2:
        C = 1
        src3 = src_img[:, :, None]
    elif src_img.ndim == 3:
        C = src_img.shape[2]
        src3 = src_img
    else:
        raise ValueError(f"unsupported image shape {src_img.shape}")
    H, W = src_img.shape[:2]
    hole_in = as_bool(hole_mask)
    if hole_in.shape != (H, W):
        raise ValueError(f"hole_mask shape {hole_in.shape} != image {H, W}")
    depth_in = np.asarray(depth, np.float32)
    if depth_in.shape != (H, W):
        raise ValueError(f"depth shape {depth_in.shape} != image {H, W}")

    ps = int(patch_size)
    if ps < 3 or ps % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")
    # adaptive-size cascade: odd sizes, largest first, always ending at `ps`
    if sizes:
        size_list = []
        for s in sizes:
            s = int(s)
            if s >= 3 and s % 2 == 1 and s not in size_list:
                size_list.append(s)
        size_list.sort(reverse=True)
    else:
        size_list = [ps]
    if ps not in size_list:
        size_list.append(ps)
    size_list = [s for s in size_list if s <= ps] or [ps]
    r = ps // 2                 # the base radius: padding / priority / pyramid geometry
    rmax = max(size_list) // 2  # the largest radius the cascade may touch
    Hp, Wp = H + 2 * r, W + 2 * r

    def _pad2(a, fill=0):
        out = np.full((Hp, Wp), fill, np.float32)
        out[r:r + H, r:r + W] = a
        return out

    def _pad3(a, fill=0):
        out = np.full((Hp, Wp, a.shape[2]), fill, np.float32)
        out[r:r + H, r:r + W] = a
        return out

    img = _pad3(src3.astype(np.float32))
    sq = (img ** 2).sum(-1).astype(np.float32)          # sum over channels of I^2
    gray = _pad2(luminance(src3))
    depth = _pad2(depth_in)
    if layer_depth is None:
        layer_map = depth
        layer_dvalid = depth > 0
    else:
        layer_map = _pad2(np.asarray(layer_depth, np.float32))
        layer_dvalid = layer_map > 0
    hole = np.zeros((Hp, Wp), bool)
    hole[r:r + H, r:r + W] = hole_in
    # the padding ring (r px) is NOT a source: it must never be used as a candidate patch
    # nor counted as a known neighbour, otherwise candidates/copies would come from
    # outside the image.
    valid0 = np.zeros((Hp, Wp), bool)
    valid0[r:r + H, r:r + W] = ~hole_in
    known = valid0.copy()                               # source region grows while filling
    dvalid = depth > 0
    Cconf = np.zeros((Hp, Wp), np.float32)
    Cconf[known] = 1.0                                  # Criminisi confidence init

    # ---- eq. (8): foreground classification (Laplacian of the predicted depth) ---- #
    if fg_mask is None and not use_bg_term:
        fg_full = np.zeros((H, W), bool)          # B == 1: classification not needed
    elif fg_mask is None:
        fg_full, _lap = laplacian_fg(depth_in, dilate=fg_dilate, valid=depth_in > 0)
    else:
        fg_full = as_bool(fg_mask)
        if fg_full.shape != (H, W):
            raise ValueError(f"fg_mask shape {fg_full.shape} != image {H, W}")
    fg = np.zeros((Hp, Wp), bool)
    fg[r:r + H, r:r + W] = fg_full
    # A LOCAL foreground mask used only by `template_exclude_fg`: pixels that are clearly
    # nearer than the depth the occlusion layer is being filled at.  The global `fg_full` mask
    # (Laplacian sign + dilation) is the right thing for the B(p) priority term, but it marks a
    # whole band around every depth step, which is too coarse to decide which TEMPLATE pixels
    # are contaminated.  Here we use the depth values directly: the separator is the midpoint
    # between the background level (background/short-run depths, i.e. the DENSE side of the
    # depth histogram) and the foreground level (the top decile of nonzero depths).
    nz = depth_in[depth_in > 0]
    if nz.size and use_bg_term:
        fg_level = float(np.percentile(nz, 90))
        bg_level = float(np.median(nz))
        sep = 0.5 * (fg_level + bg_level)
    else:
        sep = 1e9
    fg_local = np.zeros((Hp, Wp), bool)
    d_pad = depth[r:r + H, r:r + W]
    fg_local[r:r + H, r:r + W] = (d_pad > 0) & (d_pad > sep)

    # ---- eq. (7) normalisation over the whole depth map ---- #
    nzv = depth_in[depth_in > 0]
    dmax = float(nzv.max()) if nzv.size else 1.0
    dmin = float(nzv.min()) if nzv.size else 0.0
    dd = max(dmax - dmin, 1e-6)

    patch_ok = _patch_ok(valid0, ps)
    # max inverse depth inside every patch window (precomputed once): used by the
    # "same depth layer" guard, which rejects candidates touching a nearer layer
    depth_max = cv2.dilate(depth, np.ones((ps, ps), np.float32))
    # gray used for the data term: hole pixels take the nearest valid luminance
    gray_cur = _pad2(_nearest_valid_fill(luminance(src3), hole_in))
    nearest_ok = None
    if hole_in.any():
        nearest_ok = _warp.nearest_valid_index(~hole_in)   # (iy, ix) in image coords

    n_hole = int(hole_in.sum())
    rng = np.random.default_rng(seed) if seed is not None else None   # unused: deterministic

    # ------------------------------------------------------------------ #
    # priority, eq. (6)-(8)
    # ------------------------------------------------------------------ #
    off = np.arange(-r, r + 1)
    f_known = known.astype(np.float32)

    def priority_of(coords):
        """coords (N,2) padded (y,x) -> (P, C, Z) for eq. (6)."""
        ys = coords[:, 0]
        xs = coords[:, 1]
        yy = ys[:, None, None] + off[None, :, None]
        xx = xs[:, None, None] + off[None, None, :]
        kn = known[yy, xx]                                    # (N,ps,ps)
        cv = (Cconf[yy, xx] * kn).sum(axis=(1, 2)) / float(ps * ps)   # C(p)
        dv = dvalid[yy, xx] & kn
        dsum = (depth[yy, xx] * dv).sum(axis=(1, 2))
        dcnt = np.maximum(dv.sum(axis=(1, 2)), 1)
        if use_depth_term:
            zv = (dmax - dsum / dcnt) / dd                    # eq. (7)
        else:
            zv = np.ones(ys.shape[0], np.float32)
        gx = 0.5 * (gray_cur[ys, xs + 1] - gray_cur[ys, xs - 1])
        gy = 0.5 * (gray_cur[ys + 1, xs] - gray_cur[ys - 1, xs])
        kx = 0.5 * (known[ys, xs + 1].astype(np.float32) - known[ys, xs - 1])
        ky = 0.5 * (known[ys + 1, xs].astype(np.float32) - known[ys - 1, xs])
        nn = np.hypot(kx, ky)
        nx = kx / np.maximum(nn, 1e-6)
        ny = ky / np.maximum(nn, 1e-6)
        dterm = np.abs(-gy * nx + gx * ny) / float(alpha)     # D(p), isophote . normal
        if use_bg_term:
            bterm = np.where(fg[ys, xs], 0.0, 1.0).astype(np.float32)   # eq. (8)
        else:
            bterm = np.ones(ys.shape[0], np.float32)
        return (cv * dterm * zv * bterm).astype(np.float32), cv.astype(np.float32), zv

    # ------------------------------------------------------------------ #
    # fill front + lazy priority heap
    # ------------------------------------------------------------------ #
    nbf = cv2.dilate(known.astype(np.uint8), _CROSS3,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    front = hole & nbf
    prio = np.zeros((Hp, Wp), np.float32)
    stamp = np.zeros((Hp, Wp), np.int64)
    heap = []
    ver = 0

    def push_prio(coords):
        nonlocal ver
        if coords.shape[0] == 0:
            return
        pv, _cv, _zv = priority_of(coords)
        ver += 1
        ys = coords[:, 0]
        xs = coords[:, 1]
        prio[ys, xs] = pv
        stamp[ys, xs] = ver
        for k in range(coords.shape[0]):
            heapq.heappush(heap, (-float(pv[k]), int(ys[k]), int(xs[k]), ver))

    fy, fx = np.nonzero(front)
    push_prio(np.stack([fy, fx], 1))
    priority_at_first = np.zeros((H, W), np.float32)
    if fy.size:
        priority_at_first[fy - r, fx - r] = prio[fy, fx]
    fg_first_n = int(fg[fy, fx].sum()) if fy.size else 0
    fg_first_max = float(prio[fy, fx][fg[fy, fx]].max()) if fg_first_n else 0.0

    def pop_best():
        while heap:
            neg_p, y, x, v = heapq.heappop(heap)
            if front[y, x] and stamp[y, x] == v:
                return y, x, -neg_p
        return None

    # ------------------------------------------------------------------ #
    # matching helper, eq. (9)-(10)
    # ------------------------------------------------------------------ #
    def correlate(cy0, cy1, cx0, cx1, maskf, tpm, A, rr=None):
        """Masked SSD + candidate depth means for centres [cy0:cy1, cx0:cx1].

        `rr` is the patch radius actually used (defaults to the base `r`); the adaptive-size
        cascade calls this with smaller radii.
        """
        rr = r if rr is None else int(rr)
        ry0 = max(0, cy0 - rr)
        ry1 = min(Hp, cy1 + rr)
        rx0 = max(0, cx0 - rr)
        rx1 = min(Wp, cx1 + rr)
        oy = cy0 - ry0
        ox = cx0 - rx0
        y1o = oy + (cy1 - cy0)
        x1o = ox + (cx1 - cx0)
        sl = (slice(oy, y1o), slice(ox, x1o))
        bq = np.zeros((cy1 - cy0, cx1 - cx0), np.float32)
        for c in range(C):
            bq += cv2.filter2D(img[ry0:ry1, rx0:rx1, c], cv2.CV_32F,
                               tpm[:, :, c])[sl[0], sl[1]]
        cq = cv2.filter2D(sq[ry0:ry1, rx0:rx1], cv2.CV_32F, maskf)[sl[0], sl[1]]
        dq = cv2.filter2D(depth[ry0:ry1, rx0:rx1], cv2.CV_32F, maskf)[sl[0], sl[1]]
        return A - 2.0 * bq + cq, dq

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    n_iters = 0
    n_filled = 0
    # mutable single-element accumulators (match_size is a closure and needs to add to them)
    rej_no_depth = [0]
    rej_guard = [0]
    rej_none = 0
    size_hist = {}
    branch_hist = {}
    match_cost_sum = 0.0
    match_cnt = 0
    src_map = np.full((H, W), -1, np.int32) if record_source else None
    log_fn = log if log is not None else (print if verbose else (lambda *_: None))
    half = 2 * ps

    # ------------------------------------------------------------------ #
    # matcher for one patch size (used by the adaptive-size cascade)
    # ------------------------------------------------------------------ #
    def match_size(k, py, px, cval_p):
        """Find the best admissible source patch of size k for the target centred on (py,px).

        Returns dict(chosen=(qy,qx,idx)|None, ssd, branch, cost, n_adm) or None when this
        size cannot be evaluated (empty template).
        """
        kk = k // 2
        win = (slice(py - kk, py + kk + 1), slice(px - kk, px + kk + 1))
        M = known[win]
        # Optional: drop the template pixels that look like the FOREGROUND we are standing
        # next to.  The B(p) priority term already stops the fill from *starting* on
        # foreground, but once it has advanced the query patch still straddles the silhouette,
        # and those few foreground samples then dominate the SSD: the matcher is driven to
        # reproduce the dark object edge inside the hole, which shows up as a halo along the
        # seam.  They are excluded from the template only (never from the source candidates).
        if template_exclude_fg:
            fgm_t = fg_local[win]
            if fgm_t.any():
                keep = M & ~fgm_t
                if int(keep.sum()) >= max(9, int(0.25 * int(M.sum()))):
                    M = keep
        nM = int(M.sum())
        if nM == 0:
            return None
        maskf = M.astype(np.float32)
        tp_img = img[win]
        tpm = np.where(M[:, :, None], tp_img, 0.0)
        A = float((tpm * tp_img).sum())
        Zp = float((depth[win] * M).sum()) / nM

        if local_search:
            cy0 = max(kk, py - search_h // 2)
            cy1 = min(H + kk, py + search_h // 2 + 1)
            cx0 = max(kk, px - search_w // 2)
            cx1 = min(W + kk, px + search_w // 2 + 1)
        else:
            cy0, cy1, cx0, cx1 = kk, H + kk, kk, W + kk
        ssd, dq = correlate(cy0, cy1, cx0, cx1, maskf, tpm, A, rr=kk)
        zq = dq / float(nM)
        ok = patch_ok[cy0:cy1, cx0:cx1]
        ncol = cx1 - cx0
        pmax = depth_max[cy0:cy1, cx0:cx1]

        # eq. (9)-(10): the depth layer being filled is the PREDICTED background depth of
        # the hole part of the patch (layer_ref="hole", see the deviation note) -- taking
        # the mean over the valid part lets the surviving foreground drag Z_p up to the
        # foreground layer and makes DD actively prefer foreground candidates.
        if layer_ref == "hole":
            hw = hole[win]
            dvh = hw & layer_dvalid[win]
            ndvh = int(dvh.sum())
            zpl = float((layer_map[win] * dvh).sum()) / ndvh if ndvh else Zp
            if zpl <= 0:
                zpl = Zp
        else:
            zpl = Zp

        tol = float(max(depth_tol, 1e-6))
        use_guard = bool(layer_guard and use_depth_limit)
        guard = (pmax <= zpl * (1.0 + 3.0 * tol)) if use_guard else ok
        attempts = []
        if use_depth_limit:
            ddv = np.abs(zq - zpl)
            rej_no_depth[0] += int((ok & ~(ddv <= tol * zpl)).sum())
            rej_guard[0] += int((ok & (ddv <= tol * zpl) & ~guard).sum())
            attempts.append((ok & (ddv <= tol * zpl) & guard, 0))
            attempts.append((ok & (ddv <= 2.0 * tol * zpl) & guard, 1))
            attempts.append((ok & (ddv <= 4.0 * tol * zpl) & guard, 2))
            attempts.append((ok & (ddv <= 8.0 * tol * zpl) & guard, 3))
            attempts.append((ok & guard, 4))
            if use_guard:
                attempts.append((ok & (ddv <= tol * zpl), 5))
            attempts.append((ok.copy(), 6))
        else:
            attempts.append((ok.copy(), 6))

        # cross-row structural penalty: charge a vertical offset of dy pixels by
        # struct_pen * w * dy^2 per compared pixel (see the docstring)
        if struct_pen > 0.0:
            if struct_pen_w is None:
                pass
            elif struct_pen_w > 0.0:
                rows = (cy0 + np.arange(cy1 - cy0, dtype=np.float32)) - float(py)
                ssd = ssd + (struct_pen * struct_pen_w) * (rows ** 2)[:, None] * \
                    (float(C) * float(nM))

        def pick(adm):
            if not adm.any():
                return None
            idx = int(np.argmin(np.where(adm, ssd, np.inf)))
            return cy0 + idx // ncol, cx0 + idx % ncol, idx

        best = None
        for adm, br in attempts:
            c = pick(adm)
            if c is not None:
                best = dict(chosen=c, ssd=ssd, branch=br, ncol=ncol,
                            cost=float(ssd.ravel()[c[2]]) / (C * max(1, nM)),
                            n_adm=int(adm.sum()))
                break
        if best is None:
            # global search (the paper's "global -> local" starting point), no DD limit
            ssd_g, _dq_g = correlate(kk, H + kk, kk, W + kk, maskf, tpm, A, rr=kk)
            ok_g = patch_ok
            idx = int(np.argmin(np.where(ok_g, ssd_g, np.inf)))
            if np.isfinite(ssd_g.ravel()[idx]) and ok_g.ravel()[idx]:
                best = dict(chosen=(kk + idx // W, kk + idx % W, None), ssd=ssd_g, branch=7,
                            ncol=W,
                            cost=float(ssd_g.ravel()[idx]) / (C * max(1, nM)), n_adm=int(ok_g.sum()))
        return best

    # structural weight around the hole (constant for this iteration)
    struct_pen_w = 0.0

    while n_iters < max_iters:
        got = pop_best()
        if got is None:
            break
        py, px, _pp = got
        win = (slice(py - r, py + r + 1), slice(px - r, px + r + 1))
        M = known[win]
        nM = int(M.sum())
        if nM == 0:                     # isolated hole pixel: nothing to match against
            break
        # C(p) of eq. (6), evaluated *before* anything is written (Criminisi update rule)
        cval_p = float(priority_of(np.array([[py, px]]))[1][0])

        # strength of HORIZONTAL structure around the target: a big vertical gradient means
        # a rail/fence crosses here, so a vertically displaced source patch would break it
        struct_pen_w = 0.0
        if struct_pen > 0.0:
            gy0, gx0 = max(0, py - r - 7), max(0, px - r - 7)
            gwin = gray[gy0:gy0 + 15, gx0:gx0 + 15]
            if gwin.size:
                struct_pen_w = min(3.0, float(np.abs(np.diff(gwin, axis=0)).mean()) / 2.0)

        # ---- adaptive patch size ----------------------------------------------------- #
        # Two selection rules are measured against each other:
        #   "cascade" (the sibling paper's): accept the FIRST size whose best admissible match
        #       has mean SSD/channel <= beta, largest first.  Cheaper, but an absolute
        #       threshold lets a 3x3 patch (which almost always finds *some* low-cost match)
        #       pre-empt the 9x9 in far more iterations than it should.
        #   "best" (default): evaluate every size and keep the one with the lowest mean
        #       SSD/channel, with a small size bonus so that a cheap 9x9 is preferred over a
        #       marginally cheaper 3x3 (which overfits noise and loses structure).
        res = None
        k_used = ps
        for k in size_list:
            cand = match_size(k, py, px, cval_p)
            if cand is None:
                continue
            score = cand["cost"] * (1.0 - size_bonus * (k / float(max(size_list))))
            if res is None or score < res_score:
                res, res_score, k_used = cand, score, k
            if size_rule == "cascade" and (cand["cost"] <= beta or k == size_list[-1]):
                res, res_score, k_used = cand, score, k
                break
        size_hist[k_used] = size_hist.get(k_used, 0) + 1
        if res is None:
            # last resort: copy the nearest valid pixel (never expected on real data)
            branch = 8
            iy, ix = nearest_ok
            fill0 = hole[win] & ~known[win]
            fyy, fxx = np.nonzero(fill0)
            wy = fyy + py - r
            wx = fxx + px - r
            iyf = iy[wy - r, wx - r]
            ixf = ix[wy - r, wx - r]
            src_col = img[iyf + r, ixf + r]
            img[wy, wx] = src_col
            sq[wy, wx] = (src_col ** 2).sum(-1)
            depth[wy, wx] = depth[iyf + r, ixf + r]
            gray_cur[wy, wx] = gray_cur[iyf + r, ixf + r]
            Cconf[wy, wx] = cval_p
            known[wy, wx] = True
            if src_map is not None:
                src_map[wy - r, wx - r] = (iyf) * W + (ixf)
            n_filled += int(fill0.sum())
        else:
            ssd = res["ssd"]
            qy, qx = res["chosen"][0], res["chosen"][1]
            iy = res["chosen"][2]
            branch = res["branch"]
            rk = k_used // 2
            if iy is not None:
                match_cost_sum += float(ssd.ravel()[iy])
            else:
                match_cost_sum += float(ssd[qy - rk, qx - rk])
            match_cnt += nM
            swin = (slice(qy - rk, qy + rk + 1), slice(qx - rk, qx + rk + 1))
            recolor = img[swin]
            redepth = depth[swin]
            regray = gray_cur[swin]
            fill = hole[win] & ~known[win]          # only still-unfilled pixels are written
            fyy, fxx = np.nonzero(fill)
            if fyy.size:
                wy = fyy + py - r
                wx = fxx + px - r
                dy_ = np.clip(fyy - rk, 0, 2 * rk)
                dx_ = np.clip(fxx - rk, 0, 2 * rk)
                sc = recolor[dy_, dx_]
                img[wy, wx] = sc
                sq[wy, wx] = (sc ** 2).sum(-1)
                depth[wy, wx] = redepth[dy_, dx_]
                gray_cur[wy, wx] = regray[dy_, dx_]
                known[wy, wx] = True
                Cconf[wy, wx] = cval_p
                n_filled += int(fyy.size)
                if src_map is not None:
                    sy = qy + dy_
                    sx = qx + dx_
                    src_map[wy - r, wx - r] = (sy - rk) * W + (sx - rk)
                # Safety net: with the smallest patch (3x3) the copy can leave a handful of
                # pixels of the target patch unwritten on degenerate geometry (measured: 16
                # of 46171 disocclusion px on the synthetic fixture, which made step 6 fail
                # its "all disocclusion pixels filled" assertion).  Those pixels are then
                # taken from the nearest known pixel of the source patch, which keeps the
                # "only sourced from the matched patch" contract intact.
                left = hole[win] & ~known[win]
                if left.any():
                    ly, lx = np.nonzero(left)
                    ky, kx = np.nonzero(known[win])
                    if ky.size:
                        dyy = ly[:, None] - ky[None, :]
                        dxx = lx[:, None] - kx[None, :]
                        j = np.argmin(dyy * dyy + dxx * dxx, axis=1)
                        for a_, b_, j_ in zip(ly, lx, j):
                            if known[py - r + a_, px - r + b_]:
                                continue
                            img[py - r + a_, px - r + b_] = recolor[ky[j_], kx[j_]]
                            sq[py - r + a_, px - r + b_] = (recolor[ky[j_], kx[j_]] ** 2).sum()
                            depth[py - r + a_, px - r + b_] = redepth[ky[j_], kx[j_]]
                            gray_cur[py - r + a_, px - r + b_] = regray[ky[j_], kx[j_]]
                            known[py - r + a_, px - r + b_] = True
                            Cconf[py - r + a_, px - r + b_] = cval_p
                            n_filled += 1
                            if src_map is not None:
                                src_map[a_, b_] = (qy - rk + ky[j_]) * W + (qx - rk + kx[j_])
        branch_hist[int(branch)] = branch_hist.get(int(branch), 0) + 1

        # ---- refresh the front / priorities only near the patch just filled ---- #
        wy0 = max(0, py - half)
        wy1 = min(Hp, py + half + 1)
        wx0 = max(0, px - half)
        wx1 = min(Wp, px + half + 1)
        ey0, ex0 = max(0, wy0 - 1), max(0, wx0 - 1)
        ey1, ex1 = min(Hp, wy1 + 1), min(Wp, wx1 + 1)
        nb = cv2.dilate(known[ey0:ey1, ex0:ex1].astype(np.uint8), _CROSS3,
                        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        nb = nb[wy0 - ey0:wy1 - ey0, wx0 - ex0:wx1 - ex0]
        kn = known[wy0:wy1, wx0:wx1]
        hw = hole[wy0:wy1, wx0:wx1]
        sub_front = hw & ~kn & nb
        front[wy0:wy1, wx0:wx1] = sub_front
        sys_, sxs_ = np.nonzero(sub_front)
        push_prio(np.stack([sys_ + wy0, sxs_ + wx0], 1))

        n_iters += 1
        if progress_every and n_iters % progress_every == 0:
            left = int((hole & ~known).sum())
            log_fn(f"  [inpaint] iter {n_iters}: filled {n_filled}/{n_hole} px, "
                   f"{left} left, {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #
    out = img[r:r + H, r:r + W]
    if C == 1:
        out = out[:, :, 0]
    out_u8 = np.clip(np.rint(out), 0, 255).astype(np.uint8) if src_img.dtype == np.uint8 \
        else out
    dout = depth[r:r + H, r:r + W]
    dout_u8 = np.clip(np.rint(dout), 0, 255).astype(np.uint8)
    unfilled = int((hole & ~known).sum())
    seconds = time.perf_counter() - t0

    if not return_meta:
        return out_u8
    return dict(
        filled=out_u8,
        filled_depth=dout_u8,
        n_iters=n_iters,
        seconds=seconds,
        priority_at_first=priority_at_first,
        branch_hist=branch_hist,
        size_hist=dict(size_hist),
        sizes_tried=list(size_list),
        struct_pen=float(struct_pen),
        candidates_rejected_no_depth=int(rej_no_depth[0]),
        candidates_rejected_none=int(rej_none),
        candidates_rejected_layer_guard=int(rej_guard[0]),
        # extras (documented, used by step5 / tests)
        n_hole=n_hole,
        n_filled=int(n_filled),
        n_unfilled=unfilled,
        fully_filled=bool(unfilled == 0),
        match_cost_mean=(match_cost_sum / match_cnt) if match_cnt else float("nan"),
        fg_front_at_first=fg_first_n,
        fg_front_priority_max_first=fg_first_max,
        dmax=dmax, dmin=dmin,
        src_map=src_map,
        fg_used=bool(use_bg_term),
        patch_size=ps, search_w=search_w, search_h=search_h,
        depth_tol=depth_tol,
    )


def inpaint_fill_only(image, hole_mask, depth, **kw):
    """Postprocessing mode of the paper III-E: B(p) == 1 ("the restriction of the
    background term is redundant"), i.e. no foreground term in the priority."""
    kw.pop("fg_mask", None)
    kw["use_bg_term"] = False
    kw.setdefault("use_depth_term", True)
    kw.setdefault("use_depth_limit", True)
    return inpaint(image, hole_mask, depth, **kw)
