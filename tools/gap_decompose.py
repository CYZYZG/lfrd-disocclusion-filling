"""Decompose the gap to the sibling: smooth vs textured background, and where the error sits.

The texture-retention measurement (tools/texture_h2h.py) says our texture is NOT worse than the
sibling's (0.75 vs 0.61 of ground truth), yet the sibling scores 1.9 dB higher on the same
warp.  So the gap must come from content PLACEMENT rather than texture.  This splits the hole
by ground-truth texture and by distance to the hole boundary to see where the gap lives.

    python tools/gap_decompose.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2 `
        --tag ba54
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lpsnr(gt, img, mask):
    n = int(np.asarray(mask, bool).sum())
    if n == 0:
        return float("nan")
    d = ((luma(gt) - luma(img)) ** 2)[mask]
    return float(10 * np.log10(255.0 ** 2 / d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    h2h = os.path.join(io_utils.run_dir("_h2h", create=False), a.tag)
    if not os.path.isdir(h2h):
        h2h = os.path.join(io_utils.run_dir("_h2h", create=False),
                           f"{a.tag}_cam{a.src_cam}{a.dst_cam}")
    print(f"{'frame':6s} {'smooth px':>9s} {'ours':>7s} {'sib':>7s} {'gap':>6s} | "
          f"{'textured px':>11s} {'ours':>7s} {'sib':>7s} {'gap':>6s} | "
          f"{'edge px':>8s} {'ours':>7s} {'sib':>7s} {'gap':>6s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        fp = os.path.join(h2h, f"{frame}_B_noOOFA", "03_filled.png")
        if not os.path.isfile(fp):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        ours = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        sib = io_utils.imread(fp)
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        g = luma(gt).astype(np.float32)
        m1 = cv2.boxFilter(g, -1, (7, 7), normalize=True)
        m2 = cv2.boxFilter(g * g, -1, (7, 7), normalize=True)
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
        tex = region & (sd > a.tex_thr)
        smooth = region & (sd <= a.tex_thr)
        # edge band = 3 px around the hole boundary
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge = region & (cv2.dilate(region.astype(np.uint8), k) > 0) & \
            (cv2.erode(region.astype(np.uint8), k) == 0)
        o_s, s_s = lpsnr(gt, ours, smooth), lpsnr(gt, sib, smooth)
        o_t, s_t = lpsnr(gt, ours, tex), lpsnr(gt, sib, tex)
        o_e, s_e = lpsnr(gt, ours, edge), lpsnr(gt, sib, edge)
        print(f"{frame:6s} {int(smooth.sum()):9d} {o_s:7.2f} {s_s:7.2f} {o_s - s_s:+6.2f} | "
              f"{int(tex.sum()):11d} {o_t:7.2f} {s_t:7.2f} {o_t - s_t:+6.2f} | "
              f"{int(edge.sum()):8d} {o_e:7.2f} {s_e:7.2f} {o_e - s_e:+6.2f}")
        rows.append((int(smooth.sum()), o_s, s_s, int(tex.sum()), o_t, s_t, o_e, s_e))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\nMEAN  平滑背景（{m(0):.0f} px）  ours {m(1):.2f}  sib {m(2):.2f}  "
              f"gap {m(1) - m(2):+.2f}")
        print(f"      纹理背景（{m(3):.0f} px）  ours {m(4):.2f}  sib {m(5):.2f}  "
              f"gap {m(4) - m(5):+.2f}")
        print(f"      边缘带         ours {m(6):.2f}  sib {m(7):.2f}  "
              f"gap {m(6) - m(7):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
