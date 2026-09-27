"""Are the "over-textured" blocks actually wrong, or is the HF indicator misleading?

tools/artifact_map.py flags a block as over-textured when HF(result) / HF(GT) > 1.6, which
sounds like an artefact.  But a block full of fine noise has high HF too, and fine noise can sit
at the correct position.  This checks whether those blocks are actually worse by PSNR/SSIM than
the blocks the map calls acceptable, which decides whether the flag is a real defect or just a
metric quirk.

    python tools/over_texture_check.py --run ba54_latest --base ba54_seq --frames 0,5
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
    ap.add_argument("--base", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,3,5,8")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    print(f"{'frame':6s} {'class':12s} {'blocks':>6s} {'PSNR':>7s} {'SSIM':>7s} "
          f"{'HF ratio':>9s} {'mean|err|':>9s}")
    tot = {}
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
        hi, hg = hf(img), hf(gt)
        gl = luma(gt)
        B = a.block
        H, W = region.shape
        classes = {"ok": [], "under": [], "over": [], "wrong": []}
        for y in range(0, H - B + 1, B):
            for x in range(0, W - B + 1, B):
                m = region[y:y + B, x:x + B]
                if m.mean() < 0.5:
                    continue
                pad = np.pad(m, ((y, H - y - B), (x, W - x - B)))
                p = metrics.psnr(img, gt, mask=pad)
                try:
                    s = metrics.ssim(img, gt, pad, True)
                except Exception:                                      # noqa: BLE001
                    s = float("nan")
                r = hi[y:y + B, x:x + B][m].mean() / max(hg[y:y + B, x:x + B][m].mean(), 1e-6)
                err = np.abs(luma(img)[y:y + B, x:x + B][m] -
                             gl[y:y + B, x:x + B][m]).mean()
                k = ("wrong" if p < 18 else
                     "under" if r < 0.6 else "over" if r > 1.6 else "ok")
                classes[k].append((p, s, r, err))
        for k, v in classes.items():
            if not v:
                continue
            arr = np.array(v, np.float64)
            print(f"{frame:6s} {k:12s} {len(v):6d} {np.nanmean(arr[:, 0]):7.2f} "
                  f"{np.nanmean(arr[:, 1]):7.4f} {np.nanmean(arr[:, 2]):9.2f} "
                  f"{np.nanmean(arr[:, 3]):9.2f}")
            tot.setdefault(k, []).extend(v)
        print()
    print("==== 汇总 ====")
    for k, v in tot.items():
        arr = np.array(v, np.float64)
        print(f"  {k:8s} {len(v):4d} 块   PSNR {np.nanmean(arr[:, 0]):6.2f}   "
              f"SSIM {np.nanmean(arr[:, 1]):.4f}   HF比 {np.nanmean(arr[:, 2]):.2f}   "
              f"平均绝对误差 {np.nanmean(arr[:, 3]):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
