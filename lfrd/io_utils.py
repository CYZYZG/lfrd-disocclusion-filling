"""I/O helpers: non-ASCII-safe image read/write (Windows), dataset location, artefact store."""
import json
import os

import cv2
import numpy as np

from .workspace import default_dataset_root

# The MSR 3D Video Ballet dataset is research-use only and is NOT redistributed with this
# repository, so its location is resolved from the environment (BALLET_DATA_ROOT) with several
# conventional fallbacks -- see lfrd/workspace.py.  Call
# `lfrd.workspace.dataset_root(required=True)` when you want a clear error instead of a
# missing-file one.
DATASET_ROOT_DEFAULT = default_dataset_root()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")


# --------------------------------------------------------------------------- #
# image io (non-ASCII path safe: numpy.fromfile + cv2.imdecode)
# --------------------------------------------------------------------------- #
def imread(path, gray=False):
    """Read an image. Returns RGB uint8 (H,W,3) or gray uint8 (H,W)."""
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f"empty file or missing: {path}")
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise IOError(f"imdecode failed: {path}")
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(path, img_rgb):
    """Write an image; internally everything is RGB, cv2 wants BGR."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    img = img_rgb
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 2:
        pass
    else:
        raise ValueError(f"unsupported shape for imwrite: {img.shape}")
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"imencode failed: {path}")
    buf.tofile(path)


# --------------------------------------------------------------------------- #
# dataset location
# --------------------------------------------------------------------------- #
def resolve_paths(root, cam, frame):
    """Locate (color_path, depth_path); supports <root>/cam{N}/... and flat <root>/..."""
    names = (f"color-cam{cam}-{frame}.jpg", f"depth-cam{cam}-{frame}.png")
    for layout in (os.path.join(root, f"cam{cam}"), root):
        cand = [os.path.join(layout, n) for n in names]
        if all(os.path.isfile(p) for p in cand):
            return tuple(cand)
    raise FileNotFoundError(f"no color/depth pair for cam{cam} {frame} under {root}")


def load_view(root, cam, frame):
    """Return dict(color=RGB uint8, depth=P uint8 inverse-depth, paths)."""
    cpath, dpath = resolve_paths(root, cam, frame)
    color = imread(cpath, gray=False)
    depth = imread(dpath, gray=True)
    if color.shape[:2] != depth.shape[:2]:
        raise ValueError(f"size mismatch {color.shape} vs {depth.shape}")
    return {"cam": cam, "frame": frame, "color": color, "depth": depth,
            "color_path": cpath, "depth_path": dpath}


def frame_name(idx):
    """0 -> 'f000'."""
    return f"f{int(idx):03d}"


# --------------------------------------------------------------------------- #
# npz workspaces
# --------------------------------------------------------------------------- #
def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# run directory layout
# --------------------------------------------------------------------------- #
STAGE_DIRS = {
    "calib": "00_calib",
    "preproc": "10_preproc",
    "warp": "20_warp",
    "class": "30_class",
    "removal": "40_removal",
    "fill": "50_fill",
    "final": "60_final",
    "eval": "eval",
    "panels": "panels",
}


def run_dir(run_name, stage=None, create=True):
    """<project>/output/<run_name>[/<stage dir>]"""
    d = os.path.join(OUTPUT_ROOT, run_name)
    if stage:
        d = os.path.join(d, STAGE_DIRS[stage])
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def save_selection(run_name, obj):
    write_json(os.path.join(run_dir(run_name), "selection.json"), obj)


def load_selection(run_name):
    return read_json(os.path.join(run_dir(run_name), "selection.json"))
