"""Would the sibling's crack DETECTION find pixels ours misses, and would filling them help?

The sibling detects cracks with a vertical line-shaped grayscale dilation of the warped depth
and a threshold (dibr/cracks.py, paper II-A), then fills them with HHF.  Our stage 2 detects
cracks at warp time instead.  If their criterion flags residual depth discontinuities that our
warp left intact (i.e. mis-covered pixels inside valid regions), porting it could pay.

    python tools/crack_detect_compare.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sibling_cracks(depth_warped, hole, length=4, lam=5):
    """Paper II-A: max-filter with a vertical/horizontal line SE, then threshold.

    The paper's H is a vertical line; for a horizontal projection the crack runs horizontally, so
    both orientations are applied (the sibling's implementation does the same).
    """
    d = np.asarray(depth_warped, np.float64).copy()
    d[hole] = -1.0                                # empty cracks behave like translucent ones
    d = d.astype(np.float32)
    se_v = np.ones((length, 1), np.uint8)
    se_h = np.ones((1, length), np.uint8)
    # OpenCV only accepts an anchor inside the kernel, and the paper's MATLAB anchor for a 4-tap
    # SE covers the offsets {-1, 0, +1, +2} -- one past OpenCV's default (0,0), which covers
    # {-2,-1,0,+1}.  A roll by one aligns the two conventions.
    dv = np.roll(cv2.dilate(d, se_v), 1, axis=0)
    dh = np.roll(cv2.dilate(d, se_h), 1, axis=1)
    return ((dv - d) >= lam) | ((dh - d) >= lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--lam", type=int, default=5)
    a = ap.parse_args()
    print(f"{'frame':6s} {'our crack':>10s} {'sib crack':>10s} {'overlap':>8s} "
          f"{'sib-only':>9s} {'them MAE':>9s} {'warp MAE':>9s} | {'both MAE':>9s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        if not os.path.isdir(w):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        gl = luma(gt)
        ours = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        cr_ours = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        wd = np.load(os.path.join(w, "warped_depth.npy")).astype(np.float64)
        cr_sib = sibling_cracks(wd, hole, lam=a.lam) & (hole == False)
        both = cr_ours & cr_sib
        sib_only = cr_sib & ~cr_ours
        m = lambda img, mask: float(np.abs(gl - luma(img))[mask].mean()) \
            if mask.any() else float("nan")
        print(f"{frame:6s} {int(cr_ours.sum()):10d} {int(cr_sib.sum()):10d} "
              f"{int(both.sum()):8d} {int(sib_only.sum()):9d} {m(ours, sib_only):9.3f} "
              f"{m(warp, sib_only):9.3f} | {m(ours, both):9.3f}")
        rows.append((int(cr_ours.sum()), int(cr_sib.sum()), int(both.sum()),
                     int(sib_only.sum()), m(ours, sib_only), m(warp, sib_only),
                     m(ours, both)))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\nMEAN  我们的裂纹 {m(0):.0f} px, 参考项目检测 {m(1):.0f} px, 重叠 {m(2):.0f}, "
              f"仅它检出 {m(3):.0f}")
        print(f"      仅它检出的像素上 MAE：我们的结果 {m(4):.3f}  vs 纯 warp {m(5):.3f}")
        print(f"      重叠像素上我们的结果 MAE {m(6):.3f}")
        print(f"\n  判断：{'它的检测确实找到我们漏掉的坏像素，值得移植' if m(4) > m(2) * 0 + 12 else '未发现额外价值（其检出像素上我们的结果已经很好）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
