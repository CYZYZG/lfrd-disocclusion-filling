"""Clean upper bound for "claim the sibling's crack pixels": per-pixel A/B, no pipeline reruns.

For each pixel of the sibling's crack set (which our stage 6 leaves as the plain warp produced),
compare the warp value against the inpainting the sibling/our postprocess would produce, and ask
which is closer to the ground truth.  The best achievable gain from claiming those pixels is the
sum over the pixels that are better filled; if even that is ~0 dB, the mechanism is not worth
porting and no amount of engineering will change that.

    python tools/crack_claim_ceiling.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2 `
        --tag ba54
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import inpaint as I, io_utils, metrics


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sibling_cracks(depth_warped, hole, length=4, lam=5):
    d = np.asarray(depth_warped, np.float64).copy()
    d[hole] = -1.0
    d = d.astype(np.float32)
    se_v = np.ones((length, 1), np.uint8)
    se_h = np.ones((1, length), np.uint8)
    return ((np.roll(cv2.dilate(d, se_v), 1, 0) - d) >= lam) | \
           ((np.roll(cv2.dilate(d, se_h), 1, 1) - d) >= lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    print(f"{'frame':6s} {'claim px':>8s} | {'warp MAE':>8s} {'fill MAE':>8s} "
          f"{'better':>7s} {'worse':>6s} | {'CEILING hole':>12s} {'actual hole':>11s}")
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
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        wd = np.load(os.path.join(w, "warped_depth.npy")).astype(np.float64)
        claim = sibling_cracks(wd, hole) & ~hole & ~cr
        # what a fill of those pixels would look like: inpaint the warp there
        dep = np.clip(np.where(wd < 0, 0, wd), 0, 255).astype(np.uint8)
        res = I.inpaint(warp, claim, dep, patch_size=9, search_w=160, search_h=120,
                        use_bg_term=False, use_depth_term=True, use_depth_limit=False,
                        return_meta=True)
        filled = res["filled"]
        e_w = np.abs(gl - luma(warp))
        e_f = np.abs(gl - luma(filled))
        better = claim & (e_f < e_w - 1e-9)
        worse = claim & (e_f > e_w + 1e-9)
        mw = float(e_w[claim].mean()) if claim.any() else np.nan
        mf = float(e_f[claim].mean()) if claim.any() else np.nan
        # ceiling: on the claimed pixels take whichever is closer to GT
        ceiling = np.array(warp, copy=True)
        ceiling[better] = filled[better]
        region = (dis | cr) & ~oofa
        p_ceil = metrics.psnr(ceiling, gt, mask=region)
        # the actual shipped result for comparison
        final = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        p_act = metrics.psnr(final, gt, mask=region)
        print(f"{frame:6s} {int(claim.sum()):8d} | {mw:8.3f} {mf:8.3f} "
              f"{int(better.sum()):7d} {int(worse.sum()):6d} | {p_ceil:12.2f} {p_act:11.2f}")
        rows.append((int(claim.sum()), mw, mf, int(better.sum()), int(worse.sum()),
                     p_ceil - p_act))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\nMEAN  认领 {m(0):.0f} px：warp MAE {m(1):.3f} -> 填充 {m(2):.3f}；"
              f"变好 {m(3):.0f}、变差 {m(4):.0f}")
        print(f"      即使按真值逐像素择优，洞区 PSNR 也只 {m(5):+.3f} dB")
        print(f"\n  判断：{'有空间，值得做' if m(5) > 0.1 else '没有空间（上界就接近 0）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
