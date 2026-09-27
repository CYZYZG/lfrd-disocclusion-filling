"""Synthetic fixtures so that every stage can be developed/tested in isolation.

Writes a complete, self-consistent set of stage artefacts into
output/<run>/10_preproc .. 50_fill using a simple synthetic scene, so that a stage module
can be exercised without waiting for the real upstream stage.

    python tools/make_fixtures.py --run fixture

The synthetic scene contains two depth layers, a rectangular foreground block that
occludes a textured background, and a hole whose left side is background and right side
is foreground -- i.e. both cases of paper eq. (4) are present.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, viz


def build(H=192, W=256):
    rng = np.random.default_rng(7)
    # background: smooth gradient + some texture
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 3), np.uint8)
    bg[..., 0] = np.clip(120 + xx * 0.3, 0, 255)
    bg[..., 1] = np.clip(90 + yy * 0.4, 0, 255)
    bg[..., 2] = 70
    for _ in range(60):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(bg, (x, y), int(rng.integers(2, 6)),
                   tuple(int(v) for v in rng.integers(90, 200, 3)), -1)
    depth = np.full((H, W), 60, np.uint8)              # background: far, small P
    # foreground block: near, large P
    fg = np.zeros((H, W), bool)
    fg[50:150, 80:150] = True
    depth[fg] = 210
    img = bg.copy()
    img[fg] = np.array([30, 40, 200], np.uint8)
    cv2.rectangle(img, (80, 50), (149, 149), (255, 255, 255), 1)
    return img, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fixture")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frame", default="f000")
    a = ap.parse_args()

    img, depth = build()
    H, W = depth.shape

    # ---- 10_preproc -------------------------------------------------- #
    d10 = io_utils.run_dir(a.run, "preproc")
    io_utils.imwrite(os.path.join(d10, "depth_pp.png"), depth)
    ghost = np.zeros((H, W), np.uint8)
    ghost[45:55, 80:150] = 255                     # a few 'corrected' pixels
    io_utils.imwrite(os.path.join(d10, "ghost_mask.png"), ghost)
    io_utils.imwrite(os.path.join(d10, "ghost_mask_raw.png"), ghost)

    # ---- 20_warp: shift the scene right, hole on the right of the block
    shift = 18
    warped = np.zeros_like(img)
    warped[:, shift:] = img[:, :-shift]
    wdepth = np.zeros_like(depth)
    wdepth[:, shift:] = depth[:, :-shift]
    hole = np.zeros((H, W), bool)
    hole[:, :shift] = True                          # OOFA (left border, open)
    hole[60:140, 150 + shift:150 + shift + 30] = True   # disocclusion right of block
    for y in range(20, 180, 37):                    # cracks
        hole[y, 40:200] = True
    d20 = io_utils.run_dir(a.run, "warp")
    io_utils.imwrite(os.path.join(d20, "warped_color.png"), warped)
    io_utils.imwrite(os.path.join(d20, "warped_color_noprep.png"), warped)
    d = wdepth.astype(np.int16)
    d[hole] = -1
    np.save(os.path.join(d20, "warped_depth.npy"), d.astype(np.int16))
    for name, m in (("hole_all", hole),
                    ("hole_crack", np.zeros_like(hole)),
                    ("hole_disocc", hole),
                    ("hole_oofa", np.zeros_like(hole))):
        io_utils.imwrite(os.path.join(d20, f"{name}.png"), (m * 255).astype(np.uint8))
    lap = cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F, ksize=3)
    wlap = np.zeros_like(lap)
    wlap[:, shift:] = lap[:, :-shift]
    np.save(os.path.join(d20, "warped_lap.npy"), wlap.astype(np.float32))
    # backward map must be derived from shift if not provided: identity-ish
    np.savez_compressed(os.path.join(d20, "backward.npz"), idx=np.full((H, W), -1, np.int32))

    io_utils.save_npz(os.path.join(d20, "warp_meta.npz"),
                      dx=np.zeros((H, W), np.float32), dy=np.zeros((H, W), np.float32),
                      hole=hole)

    viz.panel([(img, "synthetic reference"), (warped, "synthetic warp"),
               (viz.mask_overlay(warped, hole, (255, 0, 0)), "hole")],
              path=os.path.join(d20, "..", "panels", "panel_fixture.png"))
    print("fixture written under", io_utils.run_dir(a.run))


if __name__ == "__main__":
    main()
