"""Would filling the sibling's crack set help?  A controlled experiment.

Two facts drive this:

* the sibling's crack detector (paper II-A line max-filter, lam=5) flags 6161 px on our warp,
  while our stage 2 crack set is 1028 px, and the two sets are DISJOINT
  (tools/crack_detect_compare.py);
* our stage 6 leaves those 6161 px exactly as the plain warp produced them (MAE 4.527), because
  the paper's III-E asserts pixel identity wherever the warp was valid.

So the question is concrete: does filling those extra pixels -- or, more conservatively,
re-inpainting ONLY the ones that are actually worse than what a fill would give -- improve the
result?  This runs the real stage 6 with an extended residual mask and measures.

    python tools/crack_fill_probe.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics

PY = sys.executable
ROOT = r"D:\项目\空洞填补2"


def sibling_cracks(depth_warped, hole, length=4, lam=5):
    d = np.asarray(depth_warped, np.float64).copy()
    d[hole] = -1.0
    d = d.astype(np.float32)
    se_v = np.ones((length, 1), np.uint8)
    se_h = np.ones((1, length), np.uint8)
    return ((np.roll(cv2.dilate(d, se_v), 1, 0) - d) >= lam) | \
           ((np.roll(cv2.dilate(d, se_h), 1, 1) - d) >= lam)


def run_step6(base_src, run, src, dst, frame, extra_crack=None):
    wd = io_utils.run_dir(run, create=True)
    for stage in ("10_preproc", "20_warp", "30_class", "40_removal", "50_fill"):
        s = os.path.join(base_src, stage)
        if not os.path.isdir(s):
            continue
        dd = os.path.join(wd, stage)
        os.makedirs(dd, exist_ok=True)
        for nm in os.listdir(s):
            p = os.path.join(s, nm)
            if os.path.isfile(p) and os.path.getsize(p) < 24 * 1024 * 1024:
                shutil.copy2(p, os.path.join(dd, nm))
    if extra_crack is not None:
        w = os.path.join(wd, "20_warp")
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        new_cr = cr | extra_crack
        io_utils.imwrite(os.path.join(w, "hole_crack.png"),
                         (new_cr * 255).astype(np.uint8))
        io_utils.imwrite(os.path.join(w, "hole_all.png"),
                         ((hole | (extra_crack & ~cr)) * 255).astype(np.uint8))
    subprocess.run([PY, os.path.join(ROOT, "step6_render.py"), "--run", run,
                    "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    fin = os.path.join(wd, "60_final", "final.png")
    return io_utils.imread(fin) if os.path.isfile(fin) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--variant", default="all",
                    choices=["all", "bad_only"],
                    help="fill every sibling crack, or only the ones where the warp is clearly "
                         "wrong (per-pixel comparison against a first fill attempt)")
    a = ap.parse_args()
    print(f"{'frame':6s} {'sib crack':>9s} {'base':>8s} {'+fill':>8s} {'delta':>7s} | "
          f"{'extra px':>8s} {'warp MAE':>9s} {'new MAE':>8s}")
    base_v, new_v = [], []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(bl):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        w = os.path.join(bl, "20_warp")
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        wd = np.load(os.path.join(w, "warped_depth.npy")).astype(np.float64)
        sib = sibling_cracks(wd, hole) & ~hole & ~cr
        img0 = run_step6(bl, f"cf_base_{a.src_cam}{a.dst_cam}_{frame}",
                         a.src_cam, a.dst_cam, frame)
        if img0 is None:
            continue
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        p0 = metrics.psnr(img0, gt, mask=region)
        extra = sib
        if a.variant == "bad_only":
            img1 = run_step6(bl, f"cf_try_{a.src_cam}{a.dst_cam}_{frame}",
                             a.src_cam, a.dst_cam, frame, extra_crack=sib)
            if img1 is None:
                continue
            def lum(x):
                x = np.asarray(x, np.float64)
                return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
            e0 = np.abs(lum(img0) - lum(gt))
            e1 = np.abs(lum(img1) - lum(gt))
            extra = sib & (e1 < e0)          # keep only where the fill helped
        img = run_step6(bl, f"cf_{a.variant}_{a.src_cam}{a.dst_cam}_{frame}",
                        a.src_cam, a.dst_cam, frame, extra_crack=extra)
        if img is None:
            continue
        p1 = metrics.psnr(img, gt, mask=region)
        base_v.append(p0)
        new_v.append(p1)

        def lum2(x):
            x = np.asarray(x, np.float64)
            return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        mw = float(np.abs(lum2(img0) - lum2(gt))[extra].mean()) if extra.any() else np.nan
        mn = float(np.abs(lum2(img) - lum2(gt))[extra].mean()) if extra.any() else np.nan
        print(f"{frame:6s} {int(sib.sum()):9d} {p0:8.2f} {p1:8.2f} {p1 - p0:+7.2f} | "
              f"{int(extra.sum()):8d} {mw:9.3f} {mn:8.3f}")
    if base_v:
        m = lambda v: float(np.nanmean(v))
        print(f"\n  base（不填这些像素）  {m(base_v):6.2f} dB")
        print(f"  + 填充参考项目的裂纹集 {m(new_v):6.2f} dB   ({m(new_v) - m(base_v):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
