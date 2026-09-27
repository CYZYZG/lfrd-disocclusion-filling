"""Where exactly is the residual error inside the disocclusion? Per-component breakdown.

    python tools/error_anatomy.py --run q_67 --src_cam 6 --dst_cam 7 --frames 0,1
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def gray(a):
    a = np.asarray(a, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0")
    a = ap.parse_args()
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        fd = os.path.join(io_utils.run_dir(a.run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(fd, "20_warp")
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        err = np.abs(gray(ours) - gray(gt))
        n, lab, st, _ = cv2.connectedComponentsWithStats(dis.astype(np.uint8), 8)
        print(f"=== {frame}: {int(dis.sum())} disocclusion px in {n - 1} components")
        print(f"{'comp':>5s} {'area':>7s} {'bbox w x h':>12s} {'PSNR':>7s} {'RMSE':>7s} "
              f"{'err<=10 %':>9s} {'err<=25 %':>9s} {'err>50 %':>9s}")
        order = np.argsort(-st[1:, 4]) + 1
        for i in order:
            m = lab == i
            if m.sum() < 100:
                continue
            e = err[m]
            mse = float((e ** 2).mean())
            ps = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
            x, y, ww, hh, area = st[i]
            print(f"{i:5d} {area:7d} {ww:5d} x {hh:<5d} {ps:7.2f} {np.sqrt(mse):7.2f} "
                  f"{100 * (e <= 10).mean():9.1f} {100 * (e <= 25).mean():9.1f} "
                  f"{100 * (e > 50).mean():9.1f}")
        e = err[dis]
        print(f"  all: PSNR {10 * np.log10(255.0 ** 2 / (e ** 2).mean()):.2f}  "
              f"err<=10 {100 * (e <= 10).mean():.1f}%  err>50 {100 * (e > 50).mean():.1f}%")
        # spatial: is the error concentrated at the hole boundary?
        k = np.ones((3, 3), np.uint8)
        edge = dis & ~(cv2.erode(dis.astype(np.uint8), k) > 0)
        inner = dis & (cv2.erode(dis.astype(np.uint8), k, iterations=3) > 0)
        for name, m in (("boundary (1 px)", edge), ("interior (>3 px)", inner)):
            if m.any():
                ee = err[m]
                print(f"  {name:18s} {int(m.sum()):7d} px  RMSE {np.sqrt((ee ** 2).mean()):6.2f}  "
                      f"err<=10 {100 * (ee <= 10).mean():5.1f}%  err>50 "
                      f"{100 * (ee > 50).mean():5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
