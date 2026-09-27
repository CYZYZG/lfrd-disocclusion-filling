"""Measure the actual position bias: is the filled content locally SHIFTED?

"位置不对" can mean two things and they need different fixes:
  * a local shift -- the correct content is there but displaced by a few px (fixable by
    estimating and undoing the shift)
  * wrong content  -- no shift aligns it (needs different content, not a geometry fix)

For each block over the hole this correlates the result against the ground truth over a search
window and reports the offset that maximises the match, plus the correlation at zero offset.
A consistent, non-zero peak means a systematic displacement.

    python tools/shift_measure.py --run ba54_latest --frames 0,5 --block 48
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_latest")
    ap.add_argument("--base", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,5")
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--search", type=int, default=6)
    a = ap.parse_args()
    print(f"{'frame':6s} {'variant':10s} {'blocks':>7s} {'|shift| med':>12s} "
          f"{'shift (dx,dy) mode':>20s} {'PSNR base':>10s} {'PSNR shifted':>13s}")
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        gl = luma(gt)
        for tag, run in (("paper-literal", a.base), ("latest", a.run)):
            d = os.path.join(io_utils.run_dir(run, create=False),
                             f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
            w = os.path.join(d, "20_warp")
            if not os.path.isdir(w):
                continue
            img = io_utils.imread(os.path.join(d, "60_final", "final.png"))
            il = luma(img)
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            region = (dis | cr) & ~oofa
            B, S = a.block, a.search
            H, W = region.shape
            shifts, psnrs, psnrs_shift = [], [], []
            shifted = np.array(img, copy=True)
            for y in range(0, H - B + 1, B):
                for x in range(0, W - B + 1, B):
                    m = region[y:y + B, x:x + B]
                    if m.mean() < 0.55:
                        continue
                    t = gl[y:y + B, x:x + B]
                    b = il[y:y + B, x:x + B]
                    if t.std() < 3:
                        continue                        # flat: a shift is meaningless
                    # correlation over the search window (interpretable, no cv2 on big arrays)
                    tt = t - t.mean()
                    best, bs = -2.0, (0, 0)
                    for dy in range(-S, S + 1):
                        for dx in range(-S, S + 1):
                            yy, xx = y + dy, x + dx
                            if yy < 0 or xx < 0 or yy + B > H or xx + W > W:
                                continue
                            s = il[yy:yy + B, xx:xx + B]
                            ss = s - s.mean()
                            den = np.sqrt((tt * tt).sum() * (ss * ss).sum())
                            if den <= 0:
                                continue
                            c = float((tt * ss).sum() / den)
                            if c > best:
                                best, bs = c, (dx, dy)
                    shifts.append(bs)
                    psnrs.append(metrics.psnr(img, gt, mask=np.pad(
                        m, ((y, H - y - B), (x, W - x - B)))))
                    if bs != (0, 0):
                        shifted[y:y + B, x:x + B] = img[y + bs[1]:y + bs[1] + B,
                                                        x + bs[0]:x + bs[0] + B]
                        psnrs_shift.append(metrics.psnr(shifted, gt, mask=np.pad(
                            m, ((y, H - y - B), (x, W - x - B)))))
            if not shifts:
                continue
            sh = np.array(shifts)
            mag = np.hypot(sh[:, 0], sh[:, 1])
            vals, counts = np.unique(sh, axis=0, return_counts=True)
            mode = tuple(vals[int(np.argmax(counts))])
            print(f"{frame:6s} {tag:10s} {len(shifts):7d} {np.median(mag):12.2f} "
                  f"{str(mode):>20s} {np.mean(psnrs):10.2f} "
                  f"{np.mean(psnrs_shift) if psnrs_shift else float('nan'):13.2f}")
    print("\n说明：|shift| = 每个块最匹配的偏移量中位数（0 表示内容位置正确）；")
    print("      PSNR shifted = 把每块按该偏移搬回去之后的 PSNR（上界，需要知道真值才能得到）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
