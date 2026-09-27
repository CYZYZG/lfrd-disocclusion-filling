"""On the defective blocks only: is the error a SHIFT or wrong CONTENT?

tools/over_texture_check.py splits the hole blocks into good (HF ratio ~1, PSNR 25.1) and
defective (HF ratio > 1.6, PSNR 21.7).  This takes only the defective ones and asks whether
translating them aligns with the ground truth:

  best|shift| == 0 and correlation still low   -> the content is simply wrong (needs a
                                                  different source, not a geometry fix)
  best|shift| > 0                              -> the content is right but displaced, so the
                                                  fix is geometric (depth / match position)

    python tools/shift_on_defects.py --run ba54_latest --frames 0,5
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


def hf(img):
    return np.abs(cv2.Laplacian(luma(img).astype(np.float32), cv2.CV_32F, ksize=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_latest")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,3,5,8")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--search", type=int, default=8)
    a = ap.parse_args()
    print(f"{'frame':6s} {'defect':>6s} {'|shift|0':>8s} {'|shift| mean':>12s} "
          f"{'corr@0':>7s} {'corr@best':>9s} {'PSNR@0':>7s} {'PSNR@shift':>10s} "
          f"{'dx>0':>5s} {'dx<0':>5s}")
    tot = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        gl = luma(gt)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        img = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        il = luma(img)
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        hi, hg = hf(img), hf(gt)
        B, S = a.block, a.search
        H, W = region.shape
        rec = []
        for y in range(0, H - B + 1, B):
            for x in range(0, W - B + 1, B):
                m = region[y:y + B, x:x + B]
                if m.mean() < 0.5:
                    continue
                ratio = hi[y:y + B, x:x + B][m].mean() / \
                    max(hg[y:y + B, x:x + B][m].mean(), 1e-6)
                if ratio <= 1.6:
                    continue                                  # only the defective blocks
                pad = np.pad(m, ((y, H - y - B), (x, W - x - B)))
                p0 = metrics.psnr(img, gt, mask=pad)
                t = gl[y:y + B, x:x + B]
                tt = t - t.mean()
                best, bs, b_corr = -2.0, (0, 0), 0.0
                for dy in range(-S, S + 1):
                    for dx in range(-S, S + 1):
                        yy, xx = y + dy, x + dx
                        if yy < 0 or xx < 0 or yy + B > H or xx + B > W:
                            continue
                        s = il[yy:yy + B, xx:xx + B]
                        ss = s - s.mean()
                        den = np.sqrt((tt * tt).sum() * (ss * ss).sum())
                        if den <= 0:
                            continue
                        c = float((tt * ss).sum() / den)
                        if c > best:
                            best, bs = c, (dx, dy)
                den0 = np.sqrt((tt * tt).sum() * ((il[y:y+B, x:x+B] -
                                                   il[y:y+B, x:x+B].mean()) ** 2).sum())
                c0 = float((tt * (il[y:y + B, x:x + B] - il[y:y + B, x:x + B].mean())).sum()
                           / den0) if den0 > 0 else 0.0
                if bs != (0, 0):
                    shifted = np.array(img, copy=True)
                    shifted[y:y + B, x:x + B] = img[y + bs[1]:y + bs[1] + B,
                                                    x + bs[0]:x + bs[0] + B]
                    ps = metrics.psnr(shifted, gt, mask=pad)
                else:
                    ps = p0
                rec.append((bs, c0, best, p0, ps))
        if not rec:
            continue
        mag = np.array([np.hypot(*r[0]) for r in rec])
        dxs = np.array([r[0][0] for r in rec])
        m = lambda i: float(np.nanmean([r[i] for r in rec]))
        print(f"{frame:6s} {len(rec):6d} {int((mag == 0).sum()):8d} {mag.mean():12.2f} "
              f"{m(1):7.3f} {m(2):9.3f} {m(3):7.2f} {m(4):10.2f} "
              f"{int((dxs > 0).sum()):5d} {int((dxs < 0).sum()):5d}")
        tot.extend(rec)
    if tot:
        mag = np.array([np.hypot(*r[0]) for r in tot])
        m = lambda i: float(np.nanmean([r[i] for r in tot]))
        print(f"\n==== 缺陷块汇总（{len(tot)} 块）====")
        print(f"  最佳偏移为 0 的块：{int((mag == 0).sum())} / {len(tot)} "
              f"({100 * (mag == 0).mean():.0f}%)")
        print(f"  非零块的平均偏移：{mag[mag > 0].mean() if (mag > 0).any() else 0:.2f} px，"
              f"最大 {mag.max():.0f} px")
        print(f"  零偏移相关系数 {m(1):.3f}   最佳偏移相关系数 {m(2):.3f} "
              f"(差距越大说明越像'位置错')")
        print(f"  PSNR 原样 {m(3):.2f}   按最佳偏移搬回后 {m(4):.2f} "
              f"({m(4) - m(3):+.2f})  <- 这才是位置修正的上界")
        print(f"\n  判断：{'位置问题为主' if m(4) - m(3) > 1.0 else '内容问题为主（搬位置救不回来）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
