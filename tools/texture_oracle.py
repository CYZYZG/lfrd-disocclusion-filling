"""Why is the hole texture lost?  Test the mechanism with two oracles.

Hypothesis: our occlusion layer *invents* the removed region by extending the surrounding
background, which is inherently over-smooth; the sibling reproduction instead *searches the
reference image* for a background patch, so it copies real texture.  Two measurements decide
whether that is the cause and how much a texture-aware fix could recover.

  A. "hole-visible-elsewhere" ceiling
     For each disocclusion pixel, back-project the GROUND-TRUTH virtual depth into the
     reference view.  Where that lands on reference background (not on the removed band), the
     true background IS in the reference image, just at a different place -- those pixels could
     be filled with REAL texture by a matcher.  We paste the reference colour there and measure
     PSNR + texture.

  B. "reference patch-match" ceiling
     For each hole pixel take the best-matching patch from the reference image within a local
     window, matched on the geometry we already predict, and use its texture.  This is what a
     reference-search fill would do and needs no ground truth at run time.

    python tools/texture_oracle.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, warp as W


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def hf(img):
    return np.abs(cv2.Laplacian(luma(img).astype(np.float32), cv2.CV_32F, ksize=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"{'frame':6s} {'hole px':>8s} {'visible-elsewhere':>18s} {'PSNR ours':>10s} "
          f"{'PSNR A-oracle':>14s} {'tex ours':>9s} {'tex A':>7s} {'tex GT':>7s}")
    agg = {k: [] for k in ("frac", "ours", "orc", "t_ours", "t_orc", "t_gt")}
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            continue
        ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.src_cam, frame)
        gtv = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)
        gt = gtv["color"]
        gtd = gtv["depth"].astype(np.float64)
        ours = io_utils.imread(os.path.join(base, "60_final", "final.png"))
        rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"),
                              gray=True) > 0
        oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"),
                               gray=True) > 0
        region = dis & ~oofa
        ys, xs = np.nonzero(region)
        z = calib.depth_from_P(gtd[ys, xs])
        ut, vt, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), z,
                                       cams[a.dst_cam], cams[a.src_cam])
        H, Wd = gtd.shape
        ur = np.rint(ut).astype(np.int64)
        vr = np.rint(vt).astype(np.int64)
        inb = ok & (ur >= 0) & (ur < Wd) & (vr >= 0) & (vr < H)
        # "visible elsewhere" = projects into the reference AND lands outside the removed band
        vis = np.zeros(region.shape, bool)
        sel = inb.copy()
        idx = np.nonzero(sel)[0]
        outside = ~rm[vr[idx], ur[idx]]
        vis[ys[idx[outside]], xs[idx[outside]]] = True
        # A: paste the true reference colour there (this is REAL texture from the reference)
        A = np.array(ours, copy=True)
        A[vis] = ref["color"][vr[ys[idx[outside]]], ur[ys[idx[outside]]]]
        p_ours = metrics.psnr(ours, gt, mask=region)
        p_A = metrics.psnr(A, gt, mask=region)
        st = cv2.boxFilter(luma(gt).astype(np.float32), -1, (7, 7), normalize=True)
        st2 = cv2.boxFilter((luma(gt).astype(np.float32)) ** 2, -1, (7, 7), normalize=True)
        sd = np.sqrt(np.maximum(st2 - st * st, 0))
        tex = region & (sd > a.tex_thr)
        t_ours = hf(ours)[tex].mean() if tex.any() else float("nan")
        t_A = hf(A)[tex].mean() if tex.any() else float("nan")
        t_gt = hf(gt)[tex].mean() if tex.any() else float("nan")
        frac = 100.0 * vis.sum() / max(1, int(region.sum()))
        print(f"{frame:6s} {int(region.sum()):8d} {frac:17.1f}% {p_ours:10.2f} "
              f"{p_A:14.2f} {t_ours:9.1f} {t_A:7.1f} {t_gt:7.1f}")
        for k, v in (("frac", frac), ("ours", p_ours), ("orc", p_A),
                     ("t_ours", t_ours), ("t_orc", t_A), ("t_gt", t_gt)):
            agg[k].append(v)
    m = lambda k: float(np.nanmean(agg[k]))
    print(f"\nMEAN  可见于参考图其它位置的空洞像素 {m('frac'):.1f}%   "
          f"PSNR ours {m('ours'):.2f} -> A-oracle {m('orc'):.2f} ({m('orc') - m('ours'):+.2f})")
    print(f"      纹理区 HF: GT {m('t_gt'):.1f}    ours {m('t_ours'):.1f} "
          f"(保留 {m('t_ours') / m('t_gt'):.2f})   A-oracle {m('t_orc'):.1f} "
          f"(保留 {m('t_orc') / m('t_gt'):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
