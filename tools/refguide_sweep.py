"""Parameter sweep for the reference-guided fill (evaluated through the real stage 6).

    python tools/refguide_sweep.py --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, refguide

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compose(base_src, run, src, dst, frame, c, d, pred_d):
    wd = io_utils.run_dir(run, create=True)
    for stage in ("20_warp", "30_class", "40_removal"):
        s = os.path.join(base_src, stage)
        dd = os.path.join(wd, stage)
        os.makedirs(dd, exist_ok=True)
        for nm in os.listdir(s):
            p = os.path.join(s, nm)
            if os.path.isfile(p) and os.path.getsize(p) < 24 * 1024 * 1024:
                shutil.copy2(p, os.path.join(dd, nm))
    sf = os.path.join(wd, "50_fill")
    os.makedirs(sf, exist_ok=True)
    io_utils.imwrite(os.path.join(sf, "filled_occlusion.png"), c)
    io_utils.imwrite(os.path.join(sf, "filled_occlusion_depth.png"), d)
    io_utils.imwrite(os.path.join(sf, "depth_pred.png"), pred_d)
    subprocess.run([PY, os.path.join(ROOT, "step6_render.py"), "--run", run,
                    "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame],
                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                   cwd=ROOT)
    fin = os.path.join(wd, "60_final", "final.png")
    return io_utils.imread(fin) if os.path.isfile(fin) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    grid = [(0, t) for t in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)] + [(1, 4.0), (2, 4.0),
                                                                  (0, 0.0)]
    res = {g: [] for g in grid}
    base_v = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.base_run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(bl):
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        rm = io_utils.imread(os.path.join(bl, "40_removal", "removed_mask.png"), gray=True) > 0
        pred_d = io_utils.imread(os.path.join(bl, "40_removal", "depth_pred.png"), gray=True)
        lay_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        lay_d = io_utils.imread(os.path.join(bl, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        w = os.path.join(bl, "20_warp")
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        img0 = compose(bl, f"rgs_base_{a.src_cam}{a.dst_cam}_{frame}", a.src_cam, a.dst_cam,
                       frame, lay_c, lay_d, pred_d)
        base_v.append(metrics.psnr(img0, gt, mask=region) if img0 is not None else np.nan)
        row = []
        for (search, slack) in grid:
            if search == 0 and slack == 0.0:
                row.append(base_v[-1])
                res[(search, slack)].append(base_v[-1])
                continue
            rg = refguide.reference_guided_fill(
                lay_c, lay_d, ref["color"], pred_d, rm, cams, a.src_cam, a.dst_cam,
                search=search, patch=5, max_cost=48.0, depth_slack=slack)
            im = compose(bl, f"rgs_{search}_{int(slack)}_{a.src_cam}{a.dst_cam}_{frame}",
                         a.src_cam, a.dst_cam, frame, rg["color"], rg["depth"], pred_d)
            v = metrics.psnr(im, gt, mask=region) if im is not None else float("nan")
            res[(search, slack)].append(v)
            row.append(v)
        print(f"{frame:6s} " + " ".join(f"{v:6.2f}" for v in row))
    print("\n  参数组合（search, depth_slack）:")
    print("  " + " ".join(f"({s},{t:g})" for s, t in grid))
    mb = float(np.nanmean(base_v))
    print(f"\n  base {mb:6.2f} dB")
    for g in grid:
        v = float(np.nanmean(res[g]))
        print(f"  search={g[0]:<2d} slack={g[1]:<4g}  {v:6.2f} dB   ({v - mb:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
