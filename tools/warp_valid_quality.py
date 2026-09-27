"""Are the warp-valid pixels we preserve actually WRONG?  (the sibling repairs ~47k of them)

Step 6 asserts the final view is pixel-identical to the plain warp wherever the warp was valid.
The sibling detects ~47k px of "under-covered / crack-adjacent" pixels (a depth-edge criterion)
and re-inpaints them.  If those pixels are genuinely mis-covered, our invariant is costing us;
if they are fine, the sibling is just rewriting correct content and its extra work is cosmetic.

Measured per pixel against the ground truth, split by the sibling's own repair mask.

    python tools/warp_valid_quality.py --run ba54_seq --src_cam 5 --dst_cam 4 `
        --frames 0,1,2 --tag ba54
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lpsnr(gt, img, mask):
    if not mask.any():
        return float("nan")
    d = ((luma(gt) - luma(img)) ** 2)[mask]
    return float(10 * np.log10(255.0 ** 2 / d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    h2h = os.path.join(io_utils.run_dir("_h2h", create=False), a.tag)
    if not os.path.isdir(h2h):
        h2h = os.path.join(io_utils.run_dir("_h2h", create=False),
                           f"{a.tag}_cam{a.src_cam}{a.dst_cam}")
    print(f"{'frame':6s} {'valid px':>9s} | {'warp PSNR':>10s} {'final PSNR':>11s} | "
          f"{'near-hole px':>12s} {'warp':>7s} {'final':>7s} {'sibling':>8s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        fp = os.path.join(h2h, f"{frame}_B_noOOFA", "03_filled.png")
        if not os.path.isdir(w):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        final = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        valid = ~hole
        # "under-covered": valid pixels within a few px of the hole -- where a wrong sample is
        # most likely to have landed (this is the region the sibling repairs)
        import cv2
        near = (cv2.dilate(hole.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0) & valid
        sib = io_utils.imread(fp) if os.path.isfile(fp) else None
        pw, pf = lpsnr(gt, warp, valid), lpsnr(gt, final, valid)
        nw, nf = lpsnr(gt, warp, near), lpsnr(gt, final, near)
        ns = lpsnr(gt, sib, near) if sib is not None else float("nan")
        print(f"{frame:6s} {int(valid.sum()):9d} | {pw:10.2f} {pf:11.2f} | "
              f"{int(near.sum()):12d} {nw:7.2f} {nf:7.2f} {ns:8.2f}")
        rows.append((int(valid.sum()), pw, pf, int(near.sum()), nw, nf, ns))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\nMEAN  全部有效像素 {m(0):.0f} px：warp {m(1):.2f} vs final {m(2):.2f} "
              f"(必须逐像素相同)")
        print(f"      空洞邻域有效像素 {m(3):.0f} px：warp {m(4):.2f}  final {m(5):.2f}  "
              f"sibling 重填后 {m(6):.2f}")
        print(f"      => 这部分像素上，参考项目比我们的 warp 高 {m(6) - m(4):+.2f} dB；"
              f"我们的 final 因为强制不变，差 {m(5) - m(6):+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
