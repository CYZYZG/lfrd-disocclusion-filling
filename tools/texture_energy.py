"""Is the hole content losing TEXTURE?  High-frequency energy inside the disocclusion.

A blurry fill and a textured fill can have the same PSNR while looking very different, so
PSNR does not answer "背景有纹理时填得好不好".  This measures, on the disocclusion pixels:

  * high-frequency energy (mean |Laplacian| of luma), ours vs ground truth vs the plain warp
  * the same restricted to pixels where the GROUND TRUTH is textured (local std above a
    threshold), which is the case the user complains about
  * the ratio ours/GT, i.e. how much of the real texture survives

    python tools/texture_energy.py --runs ba54_seq:5:4 ba54_temporal:5:4 q_67:6:7 --frames 0,1,2
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


def hf(img):
    g = luma(img).astype(np.float32)
    return np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))


def local_std(img, k=7):
    g = luma(img).astype(np.float32)
    m = cv2.boxFilter(g, -1, (k, k), normalize=True)
    m2 = cv2.boxFilter(g * g, -1, (k, k), normalize=True)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run:src:dst")
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tex_thr", type=float, default=12.0,
                    help="GT local std above which a hole pixel counts as 'textured'")
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    print(f"{'run':16s} {'frame':6s} {'hole px':>8s} {'tex px':>7s} {'HF GT':>7s} "
          f"{'HF ours':>8s} {'ratio':>6s} | {'HF warp':>8s} | textured-only ratio")
    for spec in a.runs:
        run, s, d = spec.split(":")
        src, dst = int(s), int(d)
        for fi in frames:
            frame = io_utils.frame_name(fi)
            fd = os.path.join(io_utils.run_dir(run, create=False),
                              f"cam{src}-cam{dst}-{frame}")
            w = os.path.join(fd, "20_warp")
            fp = os.path.join(fd, "60_final", "final.png")
            if not (os.path.isfile(fp) and os.path.isdir(w)):
                continue
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
            ours = io_utils.imread(fp)
            warp = io_utils.imread(os.path.join(w, "warped_color.png"))
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            region = dis & ~oofa
            if not region.any():
                continue
            h_gt = hf(gt)[region].mean()
            h_ou = hf(ours)[region].mean()
            h_wr = hf(warp)[region].mean()
            st = local_std(gt)
            tex = region & (st > a.tex_thr)
            if tex.sum() > 50:
                t_gt = hf(gt)[tex].mean()
                t_ou = hf(ours)[tex].mean()
                t_txt = f"{t_ou / max(t_gt, 1e-6):.2f} (GT {t_gt:.1f} -> ours {t_ou:.1f})"
            else:
                t_txt = "n/a"
            print(f"{run:16s} {frame:6s} {int(region.sum()):8d} {int(tex.sum()):7d} "
                  f"{h_gt:7.2f} {h_ou:8.2f} {h_ou / max(h_gt, 1e-6):6.2f} | {h_wr:8.2f} | "
                  f"{t_txt}")
    print("\nHF = 平均 |Laplacian(luma)|；'textured-only' 只看 GT 局部方差 >",
          a.tex_thr, "的空洞像素。ratio < 1 表示纹理被抹平。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
