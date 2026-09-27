"""Is our fill's photometric error spatially varying?  (decides what correction can help)

The current photo_correct (lfrd, stage 6) removes ONE per-channel offset+contrast for the whole
filled region, estimated at the seam.  It was worth +0.52 dB.  A spatially varying correction can
only add to that if the residual offset is itself spatially structured -- if it is flat, a global
constant is already optimal and there is nothing left.

This measures the residual (result - ground truth) mean per channel on a coarse grid inside the
hole, i.e. how much level error survives the global correction, and how much of it is low-frequency.

    python tools/photo_residual.py --run best_final --src_cam 5 --dst_cam 4 --frames 0,5
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="best_final")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,5")
    ap.add_argument("--block", type=int, default=64)
    a = ap.parse_args()
    print(f"{'frame':6s} {'blocks':>7s} | {'|resid| mean':>12s} {'per-block spread':>16s} "
          f"{'low-freq part':>13s} | {'global |bias|':>13s}")
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(d):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        final = io_utils.imread(os.path.join(d, "60_final", "final.png")).astype(np.float32)
        w = os.path.join(d, "20_warp")
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        hole = (dis | cr) & ~oofa
        H, W = hole.shape
        resid = (final - gt.astype(np.float32)).mean(2)      # luma-ish residual
        # global bias after correction
        gb = float(np.abs(resid[hole].mean()))
        vals, cy, cx = [], [], []
        B = a.block
        for y in range(0, H - B + 1, B):
            for x in range(0, W - B + 1, B):
                m = hole[y:y + B, x:x + B]
                if m.mean() < 0.5:
                    continue
                vals.append(float(resid[y:y + B, x:x + B][m].mean()))
                cy.append(y + B // 2)
                cx.append(x + B // 2)
        if not vals:
            continue
        vals = np.array(vals)
        absmean = float(np.abs(vals).mean())
        spread = float(np.percentile(vals, 90) - np.percentile(vals, 10))
        # low-frequency part: how much variance a smooth fit explains (nearest-neighbour
        # interpolation of the block means, weighted by distance)
        print(f"{frame:6s} {len(vals):7d} | {absmean:12.3f} {spread:16.3f} "
              f"{'n/a':>13s} | {gb:13.3f}")
        # print the block map coarsely so the structure is visible
        grid = np.full((H // B + 1, W // B + 1), np.nan)
        for (y, x), v in zip(zip(cy, cx), vals):
            grid[y // B, x // B] = v
        print("      残差块图（行=y/块, 列=x/块）：")
        for row in grid:
            print("      " + " ".join("  . " if not np.isfinite(v) else f"{v:4.0f}" for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
