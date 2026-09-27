"""Where does our error live inside the hole: the pixels the occlusion layer covers vs not?

Stage 5 predicts the occlusion layer, but that layer only covers part of the disocclusion once
warped (BA54: 45/100k missing), and those uncovered pixels are handed to Criminisi-style
inpainting, which extends surrounding content and is smooth.  The sibling instead searches the
reference image at run time, so it never has "no sample".

If the uncovered pixels carry most of the error, then stage 5's coverage is the real limiter --
not the matching cost, not the texture.

    python tools/coverage_breakdown.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
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


def lpsnr(gt, img, mask):
    if not np.asarray(mask, bool).any():
        return float("nan")
    d = ((luma(gt) - luma(img)) ** 2)[mask]
    return float(10 * np.log10(255.0 ** 2 / d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"{'frame':6s} {'region':>7s} | {'covered':>8s} {'psnr':>7s} | "
          f"{'uncovered':>9s} {'psnr':>7s} | {'sib on uncov':>12s} | {'overall ours':>12s} "
          f"{'sib':>7s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        warp = io_utils.imread(os.path.join(base, "20_warp", "warped_color.png"))
        ours = io_utils.imread(os.path.join(base, "60_final", "final.png"))
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
        occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        r = W.warp_view(cams, a.src_cam, a.dst_cam, occ_c, occ_d)
        cov = region & ~r["hole"]
        unc = region & r["hole"]
        sib = None
        if a.tag:
            fp = os.path.join(io_utils.run_dir("_h2h", create=False), a.tag,
                              f"{frame}_B_noOOFA", "03_filled.png")
            if os.path.isfile(fp):
                sib = io_utils.imread(fp)
        print(f"{frame:6s} {int(region.sum()):7d} | {int(cov.sum()):8d} "
              f"{lpsnr(gt, ours, cov):7.2f} | {int(unc.sum()):9d} {lpsnr(gt, ours, unc):7.2f} | "
              f"{(lpsnr(gt, sib, unc) if sib is not None else float('nan')):12.2f} | "
              f"{metrics.psnr(ours, gt, mask=region):12.2f} "
              f"{(metrics.psnr(sib, gt, mask=region) if sib is not None else float('nan')):7.2f}")
        rows.append((int(region.sum()), int(cov.sum()), lpsnr(gt, ours, cov),
                     int(unc.sum()), lpsnr(gt, ours, unc),
                     lpsnr(gt, sib, unc) if sib is not None else float("nan"),
                     metrics.psnr(ours, gt, mask=region),
                     metrics.psnr(sib, gt, mask=region) if sib is not None else float("nan")))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\nMEAN  遮挡层覆盖 {m(1):.0f}/{m(0):.0f} px "
              f"({100 * m(1) / m(0):.1f}%)，其 PSNR {m(2):.2f}")
        print(f"      未覆盖 {m(3):.0f} px（{100 * m(3) / m(0):.1f}%），其 PSNR {m(4):.2f}"
              f"   —— 参考项目在同一批像素上 {m(5):.2f}")
        print(f"      整体  ours {m(6):.2f}   sibling {m(7):.2f}")
        w_unc = m(3) / m(0)
        print(f"      若未覆盖像素的质量能与覆盖像素持平（{m(2):.2f}），整体将变成 "
              f"{m(6) + 10 * np.log10(1.0 / (1 - w_unc + w_unc * 10 ** (-(m(2) - m(4)) / 10))):.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
