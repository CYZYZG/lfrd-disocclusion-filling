"""Visualisation helpers (cv2 only -- no matplotlib in this environment).

Everything here returns or writes RGB uint8 images.  The workhorse is `panel()`:
it builds a captioned horizontal montage of arbitrary images/masks so that every
pipeline stage can dump one self-explanatory PNG for visual inspection.
"""
import os

import cv2
import numpy as np

from . import io_utils

# fixed panel geometry
PANEL_H = 384           # each tile is scaled to this height
CAP_H = 26              # caption strip height
GAP = 6
BG = (24, 24, 24)
FG = (235, 235, 235)


# --------------------------------------------------------------------------- #
# conversions
# --------------------------------------------------------------------------- #
def norm_u8(x, lo=None, hi=None, valid=None):
    """Normalise an arbitrary float array to uint8 with optional valid mask."""
    x = np.asarray(x, np.float32)
    if valid is not None:
        v = np.asarray(valid, bool)
        if v.any():
            lo = float(x[v].min()) if lo is None else lo
            hi = float(x[v].max()) if hi is None else hi
    if lo is None:
        lo = float(np.nanmin(x))
    if hi is None:
        hi = float(np.nanmax(x))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def gray_to_rgb(g):
    g = np.asarray(g)
    if g.ndim == 2:
        return np.stack([g] * 3, -1)
    return g


def colorize_map(x, valid=None, cmap=cv2.COLORMAP_TURBO):
    """Normalised float map -> RGB uint8 with a colour map; invalid pixels -> black."""
    u8 = norm_u8(x, valid=valid)
    rgb = cv2.applyColorMap(u8, cmap)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if valid is not None:
        rgb[~np.asarray(valid, bool)] = 0
    return rgb


def mask_overlay(img, mask, color=(255, 0, 0), alpha=0.55, dilate=1):
    """Blend a boolean/uint8 mask onto a colour image with a flat colour."""
    img = np.asarray(img, np.uint8).copy()
    m = np.asarray(mask)
    m = (m > 0)
    if dilate > 0 and m.any():
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dilate + 1,) * 2, np.uint8)) > 0
    col = np.array(color, np.uint8)
    img[m] = (img[m].astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha
              ).astype(np.uint8)
    return img


def contour_overlay(img, mask, color=(255, 255, 0), thickness=1):
    """Draw the contour of a mask on a colour image."""
    img = np.asarray(img, np.uint8).copy()
    m = np.asarray(mask, np.uint8)
    if m.max() > 1:
        m = (m > 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, tuple(int(v) for v in color), thickness)
    return img


def diff_map(a, b, gain=3.0):
    """Amplified absolute difference of two colour images."""
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
    return np.clip(d.mean(axis=2) * gain, 0, 255).astype(np.uint8)


def label(img, text, sub=None, color=FG, bg=(0, 0, 0)):
    """Burn a caption into the top-left corner (in place on a copy)."""
    img = np.asarray(img, np.uint8).copy()
    h, w = img.shape[:2]
    scale = max(0.45, min(0.8, w / 700.0))
    cv2.putText(img, str(text), (6, int(14 + 16 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, bg, 3, cv2.LINE_AA)
    cv2.putText(img, str(text), (6, int(14 + 16 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, str(sub), (6, int(14 + 30 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.85, bg, 3, cv2.LINE_AA)
        cv2.putText(img, str(sub), (6, int(14 + 30 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.85, color, 1, cv2.LINE_AA)
    return img


def _to_tile(item, height=PANEL_H):
    """item: (image, caption) or plain image -> RGB uint8 tile of the given height."""
    if isinstance(item, tuple):
        img, cap = item[0], item[1]
    else:
        img, cap = item, None
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = norm_u8(img)
    if img.ndim == 2:
        img = gray_to_rgb(img)
    h, w = img.shape[:2]
    nw = max(1, int(round(w * height / float(h))))
    tile = cv2.resize(img, (nw, height), interpolation=cv2.INTER_AREA)
    if cap is not None:
        strip = np.full((CAP_H, nw, 3), 0, np.uint8)
        sc = max(0.4, min(0.7, nw / 420.0))
        cv2.putText(strip, str(cap), (4, int(CAP_H * 0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, FG, 1, cv2.LINE_AA)
        tile = np.vstack([tile, strip])
    return tile


def panel(items, path=None, height=PANEL_H, max_width=2400, title=None):
    """Horizontal captioned montage.  `items` = list of images or (image, caption)."""
    tiles = [_to_tile(it, height) for it in items]
    # shrink tiles if the panel would be too wide
    total = sum(t.shape[1] for t in tiles) + GAP * (len(tiles) - 1)
    if total > max_width and tiles:
        k = (max_width - GAP * (len(tiles) - 1)) / float(total)
        tiles = [_to_tile(it, max(80, int(height * k))) for it in items]
        total = sum(t.shape[1] for t in tiles) + GAP * (len(tiles) - 1)
    H = max(t.shape[0] for t in tiles)
    out = np.full((H, total, 3), BG, np.uint8)
    x = 0
    for t in tiles:
        out[0:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + GAP
    if title:
        out = np.vstack([np.full((26, out.shape[1], 3), BG, np.uint8), out])
        cv2.putText(out, str(title), (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FG, 1,
                    cv2.LINE_AA)
    if path:
        io_utils.imwrite(path, out)
    return out


def grid(items, cols, path=None, height=PANEL_H, title=None):
    """Grid montage of captioned tiles (rows of `cols`)."""
    tiles = [_to_tile(it, height) for it in items]
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = tiles[i:i + cols]
        W = sum(t.shape[1] for t in chunk) + GAP * (len(chunk) - 1)
        H = max(t.shape[0] for t in chunk)
        row = np.full((H, W, 3), BG, np.uint8)
        x = 0
        for t in chunk:
            row[0:t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + GAP
        rows.append(row)
    W = max(r.shape[1] for r in rows)
    out = np.full((sum(r.shape[0] for r in rows) + GAP * (len(rows) - 1), W, 3), BG, np.uint8)
    y = 0
    for r in rows:
        out[y:y + r.shape[0], 0:r.shape[1]] = r
        y += r.shape[0] + GAP
    if title:
        head = np.full((26, out.shape[1], 3), BG, np.uint8)
        cv2.putText(head, str(title), (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FG, 1,
                    cv2.LINE_AA)
        out = np.vstack([head, out])
    if path:
        io_utils.imwrite(path, out)
    return out


def text_card(lines, width=1000, path=None, height=None):
    """A plain text card as an image (for stats / check summaries)."""
    lines = [str(s) for s in lines]
    lh = 20
    H = height or (lh * (len(lines) + 2))
    img = np.full((H, width, 3), 255, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 20 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (20, 20, 20), 1, cv2.LINE_AA)
    if path:
        io_utils.imwrite(path, img)
    return img


def imwrite_field(path, field, valid=None):
    """Write a signed displacement field as a colour map (0 already = no motion)."""
    v = valid if valid is not None else np.ones_like(field, bool)
    rgb = colorize_map(field, valid=v, cmap=cv2.COLORMAP_TURBO)
    io_utils.imwrite(path, rgb)


def zoom(img, cx, cy, w, h, out_w=480, out_h=360, mark=True):
    """Crop-zoom helper for close-ups; draws a marker at the crop centre."""
    img = np.asarray(img, np.uint8)
    H, W = img.shape[:2]
    x0 = int(np.clip(cx - w // 2, 0, max(0, W - w)))
    y0 = int(np.clip(cy - h // 2, 0, max(0, H - h)))
    crop = img[y0:y0 + h, x0:x0 + w].copy()
    if crop.size == 0:
        crop = img
    out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    if mark:
        cv2.rectangle(out, (0, 0), (out_w - 1, out_h - 1), (255, 255, 0), 1)
    return out
