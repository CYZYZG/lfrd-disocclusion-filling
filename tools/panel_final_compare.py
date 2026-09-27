"""Final visual: paper-literal vs +temporal+OOFA on one frame, with a disocclusion zoom."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补2")
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, viz

RUN_BEST, RUN_BASE = "ba54_best", "ba54_seq"
FRAME, SRC, DST = "f000", 5, 4


def main():
    b = os.path.join(io_utils.run_dir(RUN_BEST, create=False), f"cam{SRC}-cam{DST}-{FRAME}")
    o = os.path.join(io_utils.run_dir(RUN_BASE, create=False), f"cam{SRC}-cam{DST}-{FRAME}")
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, DST, FRAME)["color"]
    fin = io_utils.imread(os.path.join(b, "60_final", "final.png"))
    fin0 = io_utils.imread(os.path.join(o, "60_final", "final.png"))
    warp = io_utils.imread(os.path.join(b, "20_warp", "warped_color.png"))
    hole = io_utils.imread(os.path.join(b, "20_warp", "hole_all.png"), gray=True) > 0
    dis = io_utils.imread(os.path.join(b, "20_warp", "hole_disocc.png"), gray=True) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(dis.astype(np.uint8), 8)
    i = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h = int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])
    cx, cy = x + w // 2, y + h // 2
    zw, zh = max(140, int(w * 2.2)), max(140, int(h * 0.5))
    z = lambda im: viz.zoom(im, cx, cy, zw, zh, 440, 330)
    p0 = metrics.psnr(fin0, gt, mask=dis)
    p1 = metrics.psnr(fin, gt, mask=dis)
    tiles = [(gt, "ground truth (cam%d)" % DST),
             (warp, "plain warp (%.2f%% holes)" % (100 * hole.mean())),
             (fin0, "paper-literal  %.2f dB" % p0),
             (fin, "temporal + OOFA  %.2f dB (+%.2f)" % (p1, p1 - p0)),
             (z(gt), "zoom: ground truth"),
             (z(warp), "zoom: plain warp"),
             (z(fin0), "zoom: paper-literal"),
             (z(fin), "zoom: temporal + OOFA")]
    out = os.path.join(io_utils.run_dir(RUN_BEST, create=True), "panels",
                       "panel_final_compare.png")
    viz.grid(tiles, cols=4, height=250, path=out,
             title=f"BA54 {FRAME}: disocclusion PSNR {p0:.2f} -> {p1:.2f} dB")
    print("written:", out)


if __name__ == "__main__":
    main()
