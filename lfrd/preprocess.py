"""Step 1 -- morphology-based depth image preprocessing (paper III-A, eq. 1-2).

The paper's ghost removal:  a ghost is a *background* pixel that carries foreground
texture (or vice versa) at a depth discontinuity.  In the (inverse) depth image the
foreground has the LARGER value, so a pixel that is contaminated with background depth
sits *below* its 4-neighbourhood; it is detected by

    E_r(u,v) = [ (d (+) L)(u,v) - d(u,v) > th ]            ... eq. (1)
    L = [[0,1,0],[1,1,1],[0,1,0]]                          ... eq. (2)

and corrected by replacing its depth with the four-neighbourhood foreground depth
(the dilation value).  Ghosts are 1-2 px wide, so the process is executed twice.

Implementation decisions (recorded because the paper leaves them open):

* ``cv2.dilate`` with the cross kernel L is exactly ``max`` over the 4-neighbourhood
  (plus the pixel itself), i.e. "the four-neighbourhood foreground depth value".
* Pixels with ``P == 0`` (undefined depth) are **never** modified, even though eq. (1)
  marks them whenever they are surrounded by foreground.  This is a hard requirement of
  the frozen spec (section 2, step 1): with the guard off, all 158 undefined pixels of
  cam5/f000 would be overwritten.
* The correction is a pure ``max``:  it can only *raise* a pixel, never lower it
  (the paper explicitly rejects the eight-neighbourhood operator because it would eat
  too much background).
* ``min_change`` guards against marking pixels that would not actually change
  (the marked set is defined as the pixels the correction *applies to*).
"""
import cv2
import numpy as np

# eq. (2)
CROSS_KERNEL = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], np.uint8)

#: 3x3 box used to dilate the marked set before measuring the local Laplacian energy
BAND_KERNEL = np.ones((3, 3), np.uint8)


def _as_depth(P):
    P = np.asarray(P)
    if P.ndim != 2:
        raise ValueError(f"depth image must be 2-D, got shape {P.shape}")
    if P.dtype != np.uint8:
        P = np.clip(P, 0, 255).astype(np.uint8)
    return P


def dilate_cross(d):
    """``d (+) L`` of eq. (1)/(2) for an integer depth image."""
    return cv2.dilate(np.asarray(d), CROSS_KERNEL)


def mark_ghosts(d, th=20, min_change=1, protect_zero=True):
    """Eq. (1): the foreground-edge pixels that may carry ghosts.

    Returns ``(er_raw, er_used, dil)``:

    * ``er_raw``  -- the pure eq. (1) marking (``d (+) L - d > th``);
    * ``er_used`` -- the subset the correction may actually touch: the raw marking
      minus undefined (``d == 0``) pixels, and requiring a real change of at least
      ``min_change``;
    * ``dil``     -- ``d (+) L`` (the replacement value).
    """
    d = np.asarray(d)
    dil = dilate_cross(d)
    delta = dil.astype(np.int32) - d.astype(np.int32)
    er_raw = delta > int(th)
    er_used = er_raw & (delta >= int(min_change))
    if protect_zero:
        er_used = er_used & (d > 0)
    return er_raw, er_used, dil


def preprocess_depth(P, th=20, rounds=2, min_change=1, return_info=False):
    """Paper III-A depth correction (eq. 1-2).

    Parameters
    ----------
    P : (H, W) uint8 inverse depth (LARGER = nearer = foreground).
    th : eq. (1) threshold (paper IV-A: 20).
    rounds : number of iterations (paper III-A: twice, ghosts are 1-2 px wide).
    min_change : minimum actual increase for a marking to be applied.
    return_info : also return the full round-by-round diagnostics.

    Returns
    -------
    P_out : (H, W) uint8, or (if ``return_info``) a dict with

        P_in, P_out, changed        -- input / output / actually modified pixels
        marked                      -- pixels corrected by the FIRST round
        marked_all                  -- cumulative pixels corrected over all rounds
        marked_raw                  -- cumulative raw eq. (1) marking (all rounds)
        er                          -- list of per-round raw eq. (1) masks
        er_used                     -- list of per-round applied masks
        delta_max                   -- list of per-round max(dil - d)
        rounds                      -- number of executed rounds

    Guarantees (asserted by step1_preprocess.py):

    1. ``P_out >= P`` everywhere and ``P_out > P`` exactly on ``marked_all``;
    2. ``P_out[P == 0] == 0``  (undefined depth is preserved);
    3. ``P_out`` stays in ``[0, 255]``.
    """
    P_in = _as_depth(P)
    rounds = max(0, int(rounds))
    d = P_in.astype(np.int16)
    er_list, er_used_list, delta_list = [], [], []
    marked_all = np.zeros(P_in.shape, bool)
    marked_raw = np.zeros(P_in.shape, bool)
    marked_first = np.zeros(P_in.shape, bool)

    for r in range(rounds):
        er_raw, er_used, dil = mark_ghosts(d, th=th, min_change=min_change,
                                          protect_zero=True)
        delta = dil.astype(np.int32) - d.astype(np.int32)
        d = np.where(er_used, dil, d).astype(np.int16)
        er_list.append(er_raw.copy())
        er_used_list.append(er_used.copy())
        delta_list.append(int(delta.max()) if delta.size else 0)
        marked_raw |= er_raw
        marked_all |= er_used
        if r == 0:
            marked_first = er_used.copy()

    P_out = np.clip(d, 0, 255).astype(np.uint8)
    if not return_info:
        return P_out
    return dict(P_in=P_in, P_out=P_out, changed=(P_out != P_in),
                marked=marked_first, marked_all=marked_all, marked_raw=marked_raw,
                er=er_list, er_used=er_used_list, delta_max=delta_list, rounds=rounds)


def ghost_stats(P, P_out, marked):
    """Diagnostics of one preprocessing execution (see step1_preprocess.py checks).

    ``lap_energy_*`` is the mean ``|cv2.Laplacian(., CV_32F, ksize=3)|`` over the
    3x3-dilated marked set (the ghost band), i.e. how much depth-discontinuity energy
    the correction removed.
    """
    P = _as_depth(P)
    P_out = _as_depth(P_out)
    marked = np.asarray(marked, bool)
    band = cv2.dilate(marked.astype(np.uint8), BAND_KERNEL) > 0
    lap_b = np.abs(cv2.Laplacian(P.astype(np.float32), cv2.CV_32F, ksize=3))
    lap_a = np.abs(cv2.Laplacian(P_out.astype(np.float32), cv2.CV_32F, ksize=3))
    if band.any():
        lap_before = float(lap_b[band].mean())
        lap_after = float(lap_a[band].mean())
    else:
        lap_before = lap_after = 0.0
    diff = P_out.astype(np.int32) - P.astype(np.int32)
    return dict(
        n_marked=int(marked.sum()),
        n_changed=int((P_out != P).sum()),
        p_inc=int((diff > 0).sum()),
        p_dec=int((diff < 0).sum()),
        n_zero_changed=int(((P == 0) & (P_out != P)).sum()),
        n_zero_input=int((P == 0).sum()),
        max_increase=int(diff.max()) if diff.size else 0,
        band_px=int(band.sum()),
        lap_energy_before=lap_before,
        lap_energy_after=lap_after,
        lap_energy_ratio=(lap_after / lap_before) if lap_before > 0 else 0.0,
    )
