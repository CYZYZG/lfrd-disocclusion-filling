"""Is the remaining error a sub-pixel misplacement, or genuinely wrong content?

If a hole is filled with the RIGHT texture but placed a few pixels off, the PSNR is bad while
the structure is correct -- that is a different failure than synthesising the wrong texture.
This script measures, on the disocclusion pixels only:

  * PSNR as is
  * PSNR after shifting our result by dx in [-4..4] px (best case over the shift)
  * the same for the baseline (direct inpainting)

A large gain from shifting means the error is dominated by placement (depth/warp accuracy);
a small gain means the content itself is wrong.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def gray(a):
    a = np.asarray(a, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def psnr(gt, img, mask):
    d = (gray(gt) - gray(img)) ** 2
    m = np.asarray(mask, bool)
    if m.ndim == 3:
        m = m[..., 0]
    d = d[m]
    return float(10 * np.log10(255.0 ** 2 / d.mean())) if d.size else float("nan")


def shift_img(img, dx):
    out = np.zeros_like(img)
    if dx > 0:
        out[:, dx:] = img[:, :-dx]
    elif dx < 0:
        out[:, :dx] = img[:, -dx:]
    else:
        return img
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run:src:dst")
    ap.add_argument("--frames", default="0,1,2")
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    print(f"{'pair':10s} {'frame':6s} {'arm':9s} {'as-is':>8s} {'best shift':>11s} "
          f"{'gain':>7s} {'best dx':>8s}")
    for spec in a.runs:
        run, s, d = spec.split(":")
        src, dst = int(s), int(d)
        for fi in frames:
            frame = io_utils.frame_name(fi)
            fd = os.path.join(io_utils.run_dir(run, create=False),
                              f"cam{src}-cam{dst}-{frame}")
            w = os.path.join(fd, "20_warp")
            if not os.path.isdir(w):
                continue
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            arms = {"ours": io_utils.imread(os.path.join(fd, "60_final", "final.png"))}
            b = os.path.join(io_utils.run_dir(run, create=False), "ablation",
                             f"cam{src}-cam{dst}-{frame}", "direct_inpaint.png")
            if os.path.isfile(b):
                arms["baseline"] = io_utils.imread(b)
            for name, img in arms.items():
                base = psnr(gt, img, dis)
                best, bdx = base, 0
                for dx in range(-4, 5):
                    if dx == 0:
                        continue
                    v = psnr(gt, shift_img(img, dx), dis)
                    if v > best:
                        best, bdx = v, dx
                print(f"{src:>3d}->{dst:<3d}  {frame:6s} {name:9s} {base:8.2f} "
                      f"{best:11.2f} {best - base:+7.2f} {bdx:+8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
