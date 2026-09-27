"""Build one master comparison panel for a frame: GT / warp / inpainting baseline / ours.

    python make_final_panel.py --run ba54_seq --frame f000

Writes output/<run>/panels/master_cam<src>-cam<dst>-f###.png with
  row 1: ground truth, plain warp, direct inpainting baseline, proposed method
  row 2: |final-GT| amplified, |baseline-GT| amplified, and 3 zoom pairs
Also writes a compact metrics card next to it.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import cli, io_utils, metrics, viz


def main():
    ap = cli.base_parser("master comparison panel")
    a = ap.parse_args()
    cfg = cli.make_config(a)
    src, dst = cfg.src_cam, cfg.dst_cam
    frame = cli.frame_of(a)
    root = io_utils.run_dir(a.run)
    fd = os.path.join(root, f"cam{src}-cam{dst}-{frame}")
    ab = os.path.join(root, "ablation", f"cam{src}-cam{dst}-{frame}")
    outdir = os.path.join(root, "panels")
    os.makedirs(outdir, exist_ok=True)

    gt = io_utils.load_view(cfg.dataset_root, dst, frame)["color"]
    ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
    warp = io_utils.imread(os.path.join(fd, "20_warp", "warped_color.png"))
    base_p = os.path.join(ab, "direct_inpaint.png")
    base = io_utils.imread(base_p) if os.path.isfile(base_p) else None
    hole = io_utils.imread(os.path.join(fd, "20_warp", "hole_all.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(fd, "20_warp", "hole_oofa.png"), gray=True) > 0
    disocc = io_utils.imread(os.path.join(fd, "20_warp", "hole_disocc.png"), gray=True) > 0
    filled = hole & ~oofa

    # zoom on the largest disocclusion
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(disocc.astype(np.uint8), 8)
    i = 1 + int(np.argmax(st[1:, 4])) if n > 1 else 0
    x, y, w, h = (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]))
    cx, cy = x + w // 2, y + h // 2
    zw, zh = max(120, int(w * 2.2)), max(120, int(h * 0.55))

    def z(img):
        return viz.zoom(img, cx, cy, zw, zh, 480, 360)

    tiles = [(gt, "1 ground truth (cam%d)" % dst),
             (warp, "2 plain 3D warp (%d hole px)" % int(hole.sum())),
             (viz.diff_map(ours, gt), "4 |proposed - GT| x3")]
    if base is not None:
        tiles.insert(2, (base, "3 direct inpainting baseline [17]"))
        tiles.append((viz.diff_map(base, gt), "5 |baseline - GT| x3"))
    tiles += [(viz.mask_overlay(ours, filled, (0, 255, 0)), "6 filled regions (green)"),
              (z(gt), "zoom: GT"), (z(warp), "zoom: warp")]
    if base is not None:
        tiles.append((z(base), "zoom: baseline"))
    tiles.append((z(ours), "zoom: proposed"))
    viz.grid(tiles, cols=4, path=os.path.join(
        outdir, f"master_cam{src}-cam{dst}-{frame}.png"),
        height=300, title=f"Local foreground removal disocclusion filling — "
                          f"cam{src}->cam{dst} {frame}")

    # metrics card
    lines = [f"run {a.run}   cam{src} -> cam{dst}   {frame}",
             f"hole {int(hole.sum())} px ({100.0 * hole.mean():.2f}%) = cracks + "
             f"disocclusion {int(disocc.sum())} + OOFA {int(oofa.sum())}",
             ""]
    arms = [("plain warp", warp), ("proposed", ours)]
    if base is not None:
        arms.insert(1, ("direct inpainting [17]", base))
    lines.append(f"{'arm':24s} {'whole PSNR':>11s} {'whole SSIM':>11s} "
                 f"{'filled PSNR':>12s} {'filled SSIM':>12s}")
    for name, img in arms:
        mp = metrics.psnr(img, gt)
        ms = metrics.ssim(img, gt)
        # region PSNR: MSE over the region's own pixels
        fp = metrics.psnr(img, gt, mask=filled)
        fs = metrics.ssim(np.where(filled[..., None], img, gt), gt, mask=filled)
        lines.append(f"{name:24s} {mp:11.3f} {ms:11.4f} {fp:12.3f} {fs:12.4f}")
    lines += ["",
              "OOFA is left black by default (as in the paper's figures), which is why the",
              "whole-frame PSNR is low; the 'filled' column is the region this method owns."]
    viz.text_card(lines, width=1000,
                  path=os.path.join(outdir, f"master_cam{src}-cam{dst}-{frame}_metrics.png"))
    for l in lines:
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
