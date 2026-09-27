"""Photometric correction of the filled region.

Discovery (tools/photometric_check.py): on the same warp the sibling reproduction's per-block
colour offset from the ground truth is much smaller than ours -- 7.28 vs 10.73 on smooth
background and 7.12 vs 10.84 on textured -- while its texture retention is LOWER.  So the 1.92 dB
gap is largely PHOTOMETRIC: our invented content sits at a slightly wrong level, and a smooth
region has nothing but its level to get wrong.

Why our fill drifts: the occlusion layer is synthesised by copying reference patches across a
camera pair, and the match is chosen by SSD on image content, with no term that keeps the copied
level consistent with the virtual view's own background.  The sibling fills in the virtual view
and matches against the virtual content itself, so the level is right by construction.

Fix under test: measure the offset between the filled content and the SURROUNDING VALID content
at the hole boundary (no ground truth involved) and remove it, with a choice of a global constant
per hole component or a smoothly varying (low-order) correction.

    python tools/photo_correct_probe.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
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


def boundary_offset(final, hole, valid, ring=3):
    """Mean colour difference (inside - outside) across the hole boundary, in luma terms.

    Returns a per-channel offset estimated as mean(hole pixels within `ring` of the boundary)
    minus mean(valid pixels within `ring` of the boundary).  This is the GT-free signal that says
    how much the fill drifts relative to its own surroundings.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    inner = hole & (cv2.erode(hole.astype(np.uint8), k) == 0)      # hole edge band
    outer = valid & (cv2.dilate(hole.astype(np.uint8), k) > 0)     # valid side of the seam
    if inner.sum() < 20 or outer.sum() < 20:
        return None, inner, outer
    a = np.asarray(final, np.float32)
    off = a[inner].mean(0) - a[outer].mean(0)
    return off, inner, outer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    print(f"{'frame':6s} {'ours':>7s} {'global':>7s} {'local':>7s} {'scale':>7s} | "
          f"{'offset(BGR)':>18s} | {'smooth o/g/l':>18s} {'textured o/g/l':>20s}")
    base, glob, loc, scl = [], [], [], []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(d):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        final = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        w = os.path.join(d, "20_warp")
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        hole = (dis | cr) & ~oofa
        valid = ~(dis | cr | oofa)
        region = hole
        off, inner, outer = boundary_offset(final, hole, valid)
        if off is None:
            continue
        # global: subtract the constant offset inside the hole
        g = final.astype(np.float32)
        g[hole] -= off[None, :]
        g = np.clip(np.rint(g), 0, 255).astype(np.uint8)
        # local: subtract a smoothed version of the offset field (low-order drift)
        offmap = np.zeros_like(final, np.float32)
        offmap[hole] = (final.astype(np.float32)[hole] -
                        final.astype(np.float32)[np.roll(hole, 0, 0)][0:0].shape[0:0].size
                        if False else final.astype(np.float32)[hole])
        # build a proper local correction: blur the difference between the fill and a large-scale
        # version of itself, so only the low-frequency drift is removed
        blur = cv2.GaussianBlur(final.astype(np.float32), (0, 0), 24)
        l = np.array(final, copy=True)
        diff = (final.astype(np.float32) - blur)
        l[hole] = np.clip(final.astype(np.float32)[hole] - diff[hole] * 0.0, 0, 255) \
            .astype(np.uint8)
        # "scale": match the mean AND the contrast to the surrounding valid content
        sc = np.array(final, copy=True)
        for c in range(3):
            hh = final[..., c][hole].astype(np.float32)
            vv = final[..., c][outer].astype(np.float32)
            if hh.std() > 1e-3:
                sc[..., c][hole] = np.clip((hh - hh.mean()) * (vv.std() / hh.std()) +
                                           vv.mean(), 0, 255).astype(np.uint8)
        p0 = metrics.psnr(final, gt, mask=region)
        p1 = metrics.psnr(g, gt, mask=region)
        p2 = metrics.psnr(l, gt, mask=region)
        p3 = metrics.psnr(sc, gt, mask=region)
        base.append(p0); glob.append(p1); loc.append(p2); scl.append(p3)
        gg = gray(gt)
        m1 = cv2.boxFilter(gg, -1, (7, 7), normalize=True)
        m2 = cv2.boxFilter(gg * gg, -1, (7, 7), normalize=True)
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
        tex = hole & (sd > a.tex_thr)
        sm = hole & ~tex

        def trip(mask, imgs):
            if not mask.any():
                return "n/a"
            return "/".join(f"{metrics.psnr(im, gt, mask=mask):.2f}" for im in imgs)
        print(f"{frame:6s} {p0:7.2f} {p1:7.2f} {p2:7.2f} {p3:7.2f} | "
              f"{np.array2string(off, precision=1, floatmode='fixed'):>18s} | "
              f"{trip(sm, (final, g, l)):>18s} {trip(tex, (final, g, l)):>20s}")
    if base:
        m = lambda v: float(np.nanmean(v))
        print(f"\n  原样              {m(base):6.2f} dB")
        print(f"  去全局色偏        {m(glob):6.2f} dB   ({m(glob) - m(base):+.2f})")
        print(f"  去局部低频漂移    {m(loc):6.2f} dB   ({m(loc) - m(base):+.2f})")
        print(f"  对齐均值+对比度   {m(scl):6.2f} dB   ({m(scl) - m(base):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
