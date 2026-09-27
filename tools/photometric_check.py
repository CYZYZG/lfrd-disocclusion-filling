"""Photometric check: is the sibling's advantage in smooth regions a COLOUR/LEVEL match?

The puzzle: on the same warp the sibling is 1.92 dB ahead overall and 3.29 dB ahead on SMOOTH
background, yet its texture retention is LOWER than ours (0.61 vs 0.75 of the ground-truth
high-frequency energy, tools/texture_h2h.py).  Smooth regions have little texture to get right,
so a PSNR lead there must come from getting the LEVEL right -- a constant or low-frequency colour
offset -- which is exactly what a Poisson / gradient-domain blend fixes, and what a plain patch
copy does not.

So for each hole block, compare the mean colour offset from the ground truth:

    bias = mean(result) - mean(ground truth)

If the sibling's per-block bias is systematically smaller than ours, the remaining gap is
photometric, not geometric or content, and it is fixable cheaply.

    python tools/photometric_check.py --run ba54_seq --base ba54_latest --frames 0,5 --tag ba54
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def gray(img):
    a = np.asarray(img, np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    h2h = os.path.join(io_utils.run_dir("_h2h", create=False), a.tag)
    if not os.path.isdir(h2h):
        h2h = os.path.join(io_utils.run_dir("_h2h", create=False),
                           f"{a.tag}_cam{a.src_cam}{a.dst_cam}")
    print(f"{'frame':6s} {'class':9s} {'blocks':>6s} | {'bias ours':>10s} {'bias sib':>9s} "
          f"| {'biaszero ours':>14s} {'biaszero sib':>13s}")
    fold = {}
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        fp = os.path.join(h2h, f"{frame}_B_noOOFA", "03_filled.png")
        if not os.path.isfile(fp):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        ours = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        sib = io_utils.imread(fp)
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        g = gray(gt)
        m1 = cv2.boxFilter(g, -1, (7, 7), normalize=True)
        m2 = cv2.boxFilter(g * g, -1, (7, 7), normalize=True)
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
        B = a.block
        H, W = region.shape
        acc = {"smooth": [], "textured": []}
        for y in range(0, H - B + 1, B):
            for x in range(0, W - B + 1, B):
                m = region[y:y + B, x:x + B]
                if m.mean() < 0.5:
                    continue
                cls = "textured" if (sd[y:y + B, x:x + B][m] > a.tex_thr).mean() > 0.4 \
                    else "smooth"
                bo = float(gray(ours)[y:y + B, x:x + B][m].mean() -
                           g[y:y + B, x:x + B][m].mean())
                bs = float(gray(sib)[y:y + B, x:x + B][m].mean() -
                           g[y:y + B, x:x + B][m].mean())
                acc[cls].append((bo, bs))
        for cls in ("smooth", "textured"):
            if not acc[cls]:
                continue
            arr = np.array(acc[cls])
            # bias after removing the block's own mean offset (i.e. is it a constant offset?)
            bias0_o = float(np.abs(arr[:, 0]).mean())
            bias0_s = float(np.abs(arr[:, 1]).mean())
            print(f"{frame:6s} {cls:9s} {len(arr):6d} | {arr[:, 0].mean():10.3f} "
                  f"{arr[:, 1].mean():9.3f} | {bias0_o:14.3f} {bias0_s:13.3f}")
            fold.setdefault(cls, []).append((bias0_o, bias0_s))
    print()
    for cls, v in fold.items():
        arr = np.array(v)
        print(f"  {cls:9s} 平均 |各块亮度偏差|：ours {arr[:, 0].mean():.3f}  "
              f"sibling {arr[:, 1].mean():.3f}   "
              f"({'sibling 更贴合' if arr[:, 1].mean() < arr[:, 0].mean() else 'ours 更贴合'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
