"""Artifact map: where is the filled hole over-textured / under-textured / misplaced?

Answers "背景有纹理时填得好不好" per block.  For each 32x32 block that overlaps the hole, compare
the high-frequency energy of the result with the ground truth:

    ratio = HF(result) / HF(GT)
      < 0.6   under-textured (smoothed away)   -> orange
      > 1.6   over-textured (spurious detail)  -> purple
      0.6-1.6 acceptable

and separately mark the blocks whose PSNR inside the hole is below a threshold (misplaced
content) in red.  The output is the result image with those blocks tinted, so the failure mode
is visible at a glance.

    python tools/artifact_map.py --run ba54_latest --base ba54_seq --frames 0,3,5,8
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, viz


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def hf(img):
    return np.abs(cv2.Laplacian(luma(img).astype(np.float32), cv2.CV_32F, ksize=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_latest")
    ap.add_argument("--base", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,3,5,8")
    ap.add_argument("--block", type=int, default=32)
    a = ap.parse_args()
    outdir = os.path.join(io_utils.run_dir(a.run, create=True), "panels")
    print(f"{'frame':6s} {'blocks':>7s} {'under':>6s} {'over':>6s} {'lowPSNR':>8s} "
          f"{'ok':>5s}   (空洞覆盖的 {a.block}x{a.block} 块)")
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        img = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        h_img = hf(img)
        h_gt = hf(gt)
        B = a.block
        H, W = region.shape
        tint = np.array(img, copy=True)
        n_und = n_ovr = n_low = n_ok = 0
        for y in range(0, H - B + 1, B):
            for x in range(0, W - B + 1, B):
                m = region[y:y + B, x:x + B]
                if m.mean() < 0.5:
                    continue
                r = h_img[y:y + B, x:x + B][m].mean() / max(h_gt[y:y + B, x:x + B][m].mean(),
                                                            1e-6)
                ps = metrics.psnr(img, gt, mask=np.pad(m, ((y, H - y - B), (x, W - x - B))))
                colour = None
                if ps < 18.0:
                    colour = np.array([255, 40, 40], np.uint8)
                    n_low += 1
                elif r < 0.6:
                    colour = np.array([255, 170, 0], np.uint8)
                    n_und += 1
                elif r > 1.6:
                    colour = np.array([190, 40, 255], np.uint8)
                    n_ovr += 1
                else:
                    n_ok += 1
                if colour is not None:
                    blk = tint[y:y + B, x:x + B]
                    sub = blk[m]
                    blk[m] = (0.45 * sub + 0.55 * colour).astype(np.uint8)
        tiles = [(gt, "GT"), (img, "LATEST result"),
                 (io_utils.imread(os.path.join(io_utils.run_dir(a.base, create=False),
                                               f"cam{a.src_cam}-cam{a.dst_cam}-{frame}",
                                               "60_final", "final.png")), "paper-literal"),
                 (tint, "artifact map (红=PSNR<18 橙=纹理不足 紫=纹理过多)")]
        p = os.path.join(outdir, f"artifacts_{frame}.png")
        viz.grid(tiles, cols=4, height=260, path=p, title=f"BA54 {frame} — 空洞填补质量诊断")
        print(f"{frame:6s} {n_ok + n_und + n_ovr + n_low:7d} {n_und:6d} {n_ovr:6d} "
              f"{n_low:8d} {n_ok:5d}")
        print("        written:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
