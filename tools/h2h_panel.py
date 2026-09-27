"""Visual panel for the head-to-head: GT / warp / ours / sibling, plus the true hole region.

    python tools/h2h_panel.py --run q_67 --src_cam 6 --dst_cam 7 --frame f000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frame", default="f000")
    a = ap.parse_args()
    fd = os.path.join(io_utils.run_dir(a.run, create=False),
                      f"cam{a.src_cam}-cam{a.dst_cam}-{a.frame}")
    w = os.path.join(fd, "20_warp")
    h2h = os.path.join(io_utils.run_dir("_h2h", create=False),
                       f"{a.run}_cam{a.src_cam}{a.dst_cam}", f"{a.frame}_B_noOOFA")
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, a.frame)["color"]
    warp = io_utils.imread(os.path.join(w, "warped_color.png"))
    ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
    sib = io_utils.imread(os.path.join(h2h, "03_filled.png"))
    empty = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
    dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0

    # zoom on the largest truly-empty disocclusion
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(dis.astype(np.uint8), 8)
    i = 1 + int(np.argmax(st[1:, 4])) if n > 1 else 0
    x, y, ww, hh = (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]))
    cx, cy = x + ww // 2, y + hh // 2
    zw, zh = max(140, int(ww * 2.4)), max(140, int(hh * 0.5))

    z = lambda im: viz.zoom(im, cx, cy, zw, zh, 460, 340)
    tiles = [
        (gt, "ground truth"),
        (viz.mask_overlay(warp, empty, (255, 0, 0)), "1 plain warp + empty px (red)"),
        (ours, "2 **ours**"),
        (sib, "3 sibling viewfill"),
        (viz.diff_map(ours, gt), "4 |ours - GT| x3"),
        (viz.diff_map(sib, gt), "5 |sibling - GT| x3"),
        (z(gt), "zoom GT"),
        (z(warp), "zoom warp"),
        (z(ours), "zoom ours"),
        (z(sib), "zoom sibling"),
    ]
    out = os.path.join(io_utils.run_dir("_h2h", create=True),
                       f"{a.run}_cam{a.src_cam}{a.dst_cam}_{a.frame}_panel.png")
    viz.grid(tiles, cols=5, height=250, path=out,
             title=f"head-to-head on the SAME warp — cam{a.src_cam}->cam{a.dst_cam} "
                   f"{a.frame}")
    print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
