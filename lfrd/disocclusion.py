"""Step 2 hole typing (paper II-B / III-B): cracks vs disocclusions vs OOFA.

Three kinds of unfinished pixels appear in the warped virtual view (paper II-B):

* **cracks** -- the 1-2 px seams left by integer-pixel projection; "filled by the
  surrounding valid pixels", they are *not* a disocclusion;
* **disocclusion** -- the region that the background reveals because the foreground
  moved away; this is what the paper fills with predicted content;
* **OOFA** (out-of-field area) -- the image border strip that no reference pixel
  projects onto.

The three classes must be a partition of the hole mask (every hole pixel gets exactly
one type), because every later stage keys on them: step 2 fills cracks only, steps 3-6
operate on disocclusions, step 6 keeps OOFA unless ``--fill_oofa``.

Classification rule
-------------------
A connected hole component is

1. a **crack** if its maximum thickness (``cv2.distanceTransform``) is
   ``<= crack_max_width`` (paper: cracks are 1-2 px wide); crack wins over the
   border rule so that seams are always filled by the crack filler;
2. otherwise **OOFA** if it is open to the image border;
3. otherwise a **disocclusion**.

``min_area`` / ``min_width`` (spec section 4) do not change the partition: they define
``disocc_major``, the subset of disocclusions large/thick enough to be worth the
foreground-removal machinery (area >= ``min_area`` and ``2 * thickness >= min_width``).
Smaller enclosed remnants are still *disocclusions* (they belong to no removal target
and are cleaned by the step-6 postprocessing).

.. warning::
   ``lfrd.warp.interior_gap`` is used for the OOFA/enclosed split, but the version in the
   frozen core labels the **free space** instead of the hole mask, so the mask it returns
   as ``oofa`` never intersects the hole mask (``oofa & hole == 0``) and
   ``enclosed == hole``.  ``type_holes`` therefore validates the partition and falls back
   to the spec's own rule (hole components touching the image border are OOFA) when the
   core helper is degenerate; ``oofa_method`` in the returned dict records which rule was
   used.  The core was not modified (see the report to the lead).
"""
import cv2
import numpy as np

from . import warp

#: hole type ids used by ``hole_type_map`` / ``hole_type.png``
TYPE_VALID, TYPE_CRACK, TYPE_DISOCC, TYPE_OOFA = 0, 1, 2, 3
TYPE_NAMES = {TYPE_VALID: "valid", TYPE_CRACK: "crack", TYPE_DISOCC: "disocc",
              TYPE_OOFA: "oofa"}
#: RGB overlay colours required by the spec / task
TYPE_COLORS = {TYPE_CRACK: (255, 255, 0),      # yellow
               TYPE_DISOCC: (255, 0, 0),       # red
               TYPE_OOFA: (0, 0, 255)}         # blue


# --------------------------------------------------------------------------- #
# component table
# --------------------------------------------------------------------------- #
def _label(hole):
    """-> (n, labels, distance_transform) of the hole mask."""
    hole = np.asarray(hole, bool)
    n, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
    return n, labels, dist


def _row_widths(comp):
    """Horizontal width of a component: per-row pixel count -> (mean, max) over rows."""
    counts = comp.sum(axis=1)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0, 0.0
    return float(counts.mean()), float(counts.max())


def _boundary(comp, ksize=3):
    """1-px inner boundary of a boolean component."""
    k = np.ones((ksize, ksize), np.uint8)
    return comp & ~(cv2.erode(comp.astype(np.uint8), k) > 0)


def hole_stats(hole, labels=None):
    """Per-connected-component table of the hole mask.

    Returns a list of dicts, one per component (sorted by descending area):

        label, area, bbox=(x,y,w,h), max_thickness, mean_thickness,
        mean_width, max_width, touches_border, oob_rate

    ``max_thickness`` is the largest ``cv2.distanceTransform`` value inside the
    component.  Note that ``cv2.DIST_L2`` with the default 3x3 mask is the chamfer
    approximation (a 1 px line -> 0.955, a 2 px line -> 1.369, a 3 px line -> 1.91,
    a 20 px square -> 9.55), i.e. roughly half the width but quantised;
    ``mean_width``/``max_width`` are the exact per-row horizontal extent statistics.
    """
    hole = np.asarray(hole, bool)
    n, lab, dist = _label(hole)
    if labels is not None:
        lab = np.asarray(labels)
    h, w = hole.shape
    out = []
    for i in range(1, n):
        m = lab == i
        area = int(m.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(m)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        dv = dist[m]
        mw, xw = _row_widths(m[y0:y1 + 1, x0:x1 + 1])
        out.append(dict(
            label=int(i), area=area, bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            max_thickness=float(dv.max()), mean_thickness=float(dv.mean()),
            mean_width=mw, max_width=xw,
            touches_border=bool(x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1),
        ))
    out.sort(key=lambda d: -d["area"])
    return out


# --------------------------------------------------------------------------- #
# OOFA / enclosed split
# --------------------------------------------------------------------------- #
def border_open_labels(hole, labels):
    """Labels of the hole components that touch the image border (spec: OOFA)."""
    hole = np.asarray(hole, bool)
    h, w = hole.shape
    ids = np.unique(labels[hole])
    ids = ids[ids > 0]
    bad = []
    for i in ids:
        ys, xs = np.nonzero(labels == i)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            bad.append(int(i))
    return set(bad)


def _interior_gap_labels(hole, labels):
    """Labels of hole components that ``warp.interior_gap`` calls OOFA, or None.

    Returns ``None`` when the core helper does not produce a hole-level partition
    (see the module warning) so that the caller can fall back.
    """
    hole = np.asarray(hole, bool)
    enc, oofa = warp.interior_gap(hole)
    enc = np.asarray(enc, bool)
    oofa = np.asarray(oofa, bool)
    enc_h, oofa_h = hole & enc, hole & oofa
    if (enc_h & oofa_h).any():
        return None
    if not np.array_equal(enc_h | oofa_h, hole):
        return None
    border = border_open_labels(hole, labels)
    if bool(oofa_h.any()) != bool(border):
        return None                      # degenerate: core returned free space
    ids = np.unique(labels[oofa_h])
    return set(int(i) for i in ids if i > 0)


# --------------------------------------------------------------------------- #
# the classifier
# --------------------------------------------------------------------------- #
def type_holes(hole, crack_max_width=2, min_area=60, min_width=4, src_oob=None):
    """Split ``hole`` into crack / disocclusion / OOFA (see the module docstring).

    Parameters
    ----------
    hole : (H, W) bool (or uint8, 255 = hole).
    crack_max_width : max thickness of a component still considered a crack.
    min_area, min_width : thresholds for the ``disocc_major`` sub-mask.
    src_oob : optional (H, W) bool, True where the reference pixel projected behind the
        virtual camera; used to report the source-side out-of-bounds rate of each
        component (spec section 2, step 2) -- it does not change the partition.

    Returns
    -------
    dict with

        crack, disocc, oofa   -- disjoint bool masks whose union is ``hole``
        disocc_major          -- large/thick subset of ``disocc``
        hole_type_map         -- uint8, 0=valid 1=crack 2=disocc 3=oofa
        labels, table         -- component labels and the per-component table
        n_hole, n_crack, n_disocc, n_oofa, n_disocc_major
        oofa_method, interior_gap_ok, crack_max_width, min_area, min_width
    """
    hole = np.asarray(hole)
    if hole.dtype != bool:
        hole = hole > 0
    hole = np.ascontiguousarray(hole, bool)
    n, labels, dist = _label(hole)

    ig_labels = _interior_gap_labels(hole, labels)
    interior_gap_ok = ig_labels is not None
    if interior_gap_ok:
        oofa_labels = ig_labels
        oofa_method = "warp.interior_gap"
    else:
        oofa_labels = border_open_labels(hole, labels)
        oofa_method = "border_open_fallback (warp.interior_gap is degenerate)"

    crack = np.zeros_like(hole)
    disocc = np.zeros_like(hole)
    oofa = np.zeros_like(hole)
    table = []
    for i in range(1, n):
        m = labels == i
        area = int(m.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(m)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        thickness = float(dist[m].max())
        if thickness <= crack_max_width:
            t, why = TYPE_CRACK, "thin"
        elif i in oofa_labels:
            t, why = TYPE_OOFA, "border_open"
        else:
            t, why = TYPE_DISOCC, "enclosed"
        (crack if t == TYPE_CRACK else disocc if t == TYPE_DISOCC else oofa)[m] = True
        oob_rate = 0.0
        if src_oob is not None:
            b = _boundary(m)
            if b.any():
                oob_rate = float(np.asarray(src_oob, bool)[b].mean())
        table.append(dict(
            label=int(i), area=area, bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            max_thickness=thickness, touches_border=bool(
                x0 == 0 or y0 == 0 or x1 == hole.shape[1] - 1 or y1 == hole.shape[0] - 1),
            type=int(t), type_name=TYPE_NAMES[t], reason=why,
            major=bool(t == TYPE_DISOCC and area >= min_area
                       and 2.0 * thickness >= min_width),
            oob_rate=oob_rate,
        ))
    table.sort(key=lambda d: -d["area"])

    disocc_major = np.zeros_like(hole)
    for row, i in zip(table, [d["label"] for d in table]):
        if row["major"]:
            disocc_major |= labels == i

    hole_type_map = np.zeros(hole.shape, np.uint8)
    hole_type_map[crack] = TYPE_CRACK
    hole_type_map[disocc] = TYPE_DISOCC
    hole_type_map[oofa] = TYPE_OOFA
    return dict(
        crack=crack, disocc=disocc, oofa=oofa, disocc_major=disocc_major,
        hole_type_map=hole_type_map, hole=hole, labels=labels, table=table,
        n_hole=int(hole.sum()), n_crack=int(crack.sum()),
        n_disocc=int(disocc.sum()), n_oofa=int(oofa.sum()),
        n_disocc_major=int(disocc_major.sum()),
        n_components=len(table),
        oofa_method=oofa_method, interior_gap_ok=interior_gap_ok,
        crack_max_width=int(crack_max_width), min_area=int(min_area),
        min_width=int(min_width),
    )


# --------------------------------------------------------------------------- #
# visualisation helpers (used by step2_warp.py)
# --------------------------------------------------------------------------- #
def type_overlay(img, typing, alpha=0.6):
    """Blend the colour-coded crack/disocc/oofa masks onto a colour image."""
    out = np.asarray(img, np.uint8).copy()
    tmap = typing["hole_type_map"] if isinstance(typing, dict) else np.asarray(typing)
    for t in (TYPE_CRACK, TYPE_DISOCC, TYPE_OOFA):
        m = tmap == t
        if m.any():
            col = np.array(TYPE_COLORS[t], np.float32)
            out[m] = (out[m].astype(np.float32) * (1 - alpha) + col * alpha).astype(np.uint8)
    return out


def type_color_map(typing):
    """The hole type map as an RGB image (valid = black)."""
    tmap = typing["hole_type_map"] if isinstance(typing, dict) else np.asarray(typing)
    out = np.zeros(tmap.shape + (3,), np.uint8)
    for t, col in TYPE_COLORS.items():
        out[tmap == t] = col
    return out


def largest_bbox(mask):
    """Bounding box ``(x, y, w, h)`` of the largest connected component, or None."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, w, h, _a = [int(v) for v in stats[i]]
    return (x, y, w, h)
