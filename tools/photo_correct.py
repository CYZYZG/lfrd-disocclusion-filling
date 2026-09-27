"""Apply the boundary mean+contrast match to an existing run's output and score it.

tools/photo_correct_probe.py found that matching the filled region's MEAN and CONTRAST to the
surrounding valid content at the hole boundary (a ground-truth-free signal) is worth +0.19 dB on
the paper-literal arm.  This applies the same correction post hoc to every frame of a run and
reports the gain, which is exactly equivalent to doing it inside stage 6 (the correction is a
per-channel affine map estimated from the seam and applied to the hole).

    python tools/photo_correct.py --runs ba54_seq ba54_temporal ba54_latest
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics


def gray(img):
    a = np.asarray(img, np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def correct(final, hole, valid, ring=3, clip=25.0, contrast=True):
    """Match the hole's per-channel mean (and optionally contrast) to the boundary band."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    inner = hole & (cv2.erode(hole.astype(np.uint8), k) == 0)
    outer = valid & (cv2.dilate(hole.astype(np.uint8), k) > 0)
    if inner.sum() < 20 or outer.sum() < 20:
        return np.array(final, copy=True), None
    f = final.astype(np.float32)
    out = np.array(final, copy=True)
    info = []
    for c in range(3):
        hh = f[..., c][inner]
        vv = f[..., c][outer]
        mu_h, mu_v = float(hh.mean()), float(vv.mean())
        gain = 1.0
        if contrast and hh.std() > 1e-3 and vv.std() > 1e-3:
            gain = float(np.clip(vv.std() / hh.std(), 0.8, 1.25))
        shift = float(np.clip(mu_v - gain * mu_h, -clip, clip))
        out[..., c][hole] = np.clip(gain * f[..., c][hole] + shift, 0, 255).astype(np.uint8)
        info.append((gain, shift))
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--ring", type=int, default=3)
    ap.add_argument("--no_contrast", action="store_true")
    a = ap.parse_args()
    out_root = os.path.join(io_utils.run_dir("_photo", create=True))
    for run in a.runs:
        rows = []
        for fi in [int(x) for x in a.frames.split(",")]:
            frame = io_utils.frame_name(fi)
            d = os.path.join(io_utils.run_dir(run, create=False),
                             f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
            if not os.path.isdir(d):
                continue
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
            final = io_utils.imread(os.path.join(d, "60_final", "final.png"))
            w = os.path.join(d, "20_warp")
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            if not os.path.isfile(os.path.join(w, "hole_crack.png")):
                continue
            hole = (dis | cr) & ~oofa
            valid = ~(dis | cr | oofa)
            cor, info = correct(final, hole, valid, ring=a.ring,
                                contrast=not a.no_contrast)
            p0 = metrics.psnr(final, gt, mask=hole)
            p1 = metrics.psnr(cor, gt, mask=hole)
            rows.append((frame, p0, p1))
            io_utils.imwrite(os.path.join(out_root, f"{run}_{frame}_corrected.png"), cor)
        if rows:
            m = lambda i: float(np.nanmean([r[i] for r in rows]))
            print(f"  {run:16s} n={len(rows):2d}  原样 {m(1):6.2f} -> 校正 {m(2):6.2f}  "
                  f"({m(2) - m(1):+.2f} dB)  最好 "
                  f"{max(r[2] - r[1] for r in rows):+.2f} 最差 "
                  f"{min(r[2] - r[1] for r in rows):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
