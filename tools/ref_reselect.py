"""Validate the reference-texture re-selection fix before wiring it into stage 6.

Finding: every disocclusion pixel back-projects into the reference frame (100%), and ~31% of
them land on reference BACKGROUND -- the real texture is right there, at a different position.
The pipeline still synthesises those pixels, because stage 5 only fills the removed band and
stage 6 inpaints whatever the occlusion layer does not cover, producing over-smooth content.

The fix under test:
    for each hole pixel, back-project with the composite (predicted) depth, take the reference
    pixel when it is background, keep the current value otherwise.

    python tools/ref_reselect.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0-3
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics


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
    ap.add_argument("--bg_tol", type=float, default=0.0,
                    help="extra margin (in P) added to the background separator")
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"{'frame':6s} {'hole':>7s} {'replaced':>9s} {'PSNR ours':>10s} "
          f"{'PSNR fix':>9s} {'delta':>7s} {'tex ours':>9s} {'tex fix':>8s} {'tex GT':>7s}")
    agg = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            continue
        ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.src_cam, frame)
        gtv = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)
        gt = gtv["color"]
        ours = io_utils.imread(os.path.join(base, "60_final", "final.png"))
        fg = io_utils.imread(os.path.join(base, "30_class", "fg_mask.png"), gray=True) > 0
        dref = ref["depth"]
        nzv = dref[dref > 0]
        sep = float(np.percentile(nzv, 60)) + a.bg_tol
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"),
                              gray=True) > 0
        cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"),
                             gray=True) > 0
        oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"),
                               gray=True) > 0
        region = (dis | cr) & ~oofa
        comp = np.load(os.path.join(base, "60_final", "final_depth.npy")).astype(np.float64)
        ys, xs = np.nonzero(region)
        z = comp[ys, xs]
        ut, vt, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), z,
                                       cams[a.dst_cam], cams[a.src_cam])
        H, W = gt.shape[:2]
        ur = np.rint(ut).astype(np.int64)
        vr = np.rint(vt).astype(np.int64)
        inb = ok & (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H) & (z > 0)
        ii = np.nonzero(inb)[0]
        is_bg = np.zeros(ys.size, bool)
        if ii.size:
            is_bg[ii] = (~fg[vr[ii], ur[ii]]) & (dref[vr[ii], ur[ii]] <= sep) \
                & (dref[vr[ii], ur[ii]] > 0)
        sel = np.zeros(region.shape, bool)
        jj = np.nonzero(is_bg)[0]
        sel[ys[jj], xs[jj]] = True
        fixed = np.array(ours, copy=True)
        if jj.size:
            fixed[ys[jj], xs[jj]] = ref["color"][vr[jj], ur[jj]]
        p0 = metrics.psnr(ours, gt, mask=region)
        p1 = metrics.psnr(fixed, gt, mask=region)
        st = cv2.boxFilter(luma(gt).astype(np.float32), -1, (7, 7), normalize=True)
        st2 = cv2.boxFilter((luma(gt).astype(np.float32)) ** 2, -1, (7, 7), normalize=True)
        sd = np.sqrt(np.maximum(st2 - st * st, 0))
        tex = region & (sd > a.tex_thr)
        t0 = hf(ours)[tex].mean() if tex.any() else float("nan")
        t1 = hf(fixed)[tex].mean() if tex.any() else float("nan")
        tgt = hf(gt)[tex].mean() if tex.any() else float("nan")
        print(f"{frame:6s} {int(region.sum()):7d} {int(sel.sum()):9d} {p0:10.2f} "
              f"{p1:9.2f} {p1 - p0:+7.2f} {t0:9.2f} {t1:8.2f} {tgt:7.2f}")
        agg.append((int(region.sum()), int(sel.sum()), p0, p1, t0, t1, tgt))
    if agg:
        m = lambda i: float(np.nanmean([r[i] for r in agg]))
        print(f"\nMEAN  替换 {m(1):.0f}/{m(0):.0f} px（{100 * m(1) / m(0):.1f}%）   "
              f"PSNR {m(2):.2f} -> {m(3):.2f} ({m(3) - m(2):+.2f} dB)   "
              f"纹理区 HF {m(4):.1f} -> {m(5):.1f}（GT {m(6):.1f}，保留率 "
              f"{m(4) / m(6):.2f} -> {m(5) / m(6):.2f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
