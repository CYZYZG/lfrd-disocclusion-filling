"""Colour aggregation: mean vs median vs a recent-frames weighted median.

tools/temporal_colour_check.py shows the frames the model samples differ from each other by
~8.4 units of colour (they cover a moving background / changing lighting), and the current
implementation averages them.  Averaging across inconsistent samples is exactly how invented
"wrong content" appears at the correct position -- which is what tools/shift_on_defects.py
measured (shifting the defective blocks back only recovers +0.47 dB, so it is not a shift).

This A/B compares aggregation rules and composites each variant through the real stage 6:

    mean        current behaviour
    median      robust to the outlier frames
    med-recent  median over the frames closest in time to the target
    best        the single frame whose background colour is the most consistent with its
                neighbours (least motion blur / occlusion fringe)

    python tools/temporal_agg.py --src_cam 5 --dst_cam 4 --frames 0,5
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aggregate(root, cam, frames, region, q=10.0, tol=4.0, ref_level=None, rule="mean",
              target=None, recent=None):
    ys, xs = np.nonzero(region)
    n = len(frames)
    P = np.empty((n, ys.size), np.int16)
    COL = np.empty((n, ys.size, 3), np.uint8)
    for i, f in enumerate(frames):
        P[i] = io_utils.imread(os.path.join(root, f"cam{cam}",
                                            f"depth-cam{cam}-f{f:03d}.png"),
                               gray=True)[ys, xs]
        COL[i] = io_utils.imread(os.path.join(root, f"cam{cam}",
                                              f"color-cam{cam}-f{f:03d}.jpg"))[ys, xs]
    z = np.percentile(P, q, axis=0).astype(np.float32)
    if ref_level is not None:
        lvl = np.asarray(ref_level, np.float32)[ys, xs]
        ok = np.isfinite(lvl) & (lvl > 0)
        over = ok & (z > lvl + 3.0)
        z[over] = lvl[over]
    near = (P <= (z[None, :] + tol)) & (P > 0)
    if rule == "med-recent" and recent is not None:
        lo, hi = max(0, target - recent), min(n, target + recent + 1)
        near[:lo] = False
        near[hi:] = False
    n_s = near.sum(0)
    col = np.zeros((ys.size, 3), np.float32)
    if rule == "mean":
        for i in range(n):
            sel = near[i]
            if sel.any():
                col[sel] += COL[i][sel].astype(np.float32)
        have = n_s > 0
        col[have] /= n_s[have, None]
    else:
        per = np.full((n, ys.size, 3), np.nan, np.float32)
        for i in range(n):
            per[i][near[i]] = COL[i][near[i]].astype(np.float32)
        with np.errstate(all="ignore"):
            med = np.nanmedian(per, 0)
        have = ~np.isnan(med[:, 0])
        col[have] = med[have]
    H, W = region.shape
    c = np.zeros((H, W, 3), np.uint8)
    d = np.zeros((H, W), np.uint8)
    c[ys[have], xs[have]] = np.clip(np.rint(col[have]), 0, 255).astype(np.uint8)
    d[ys[have], xs[have]] = np.clip(np.rint(z[have]), 0, 255).astype(np.uint8)
    return c, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,5")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--recent", type=int, default=15)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    variants = ["mean", "median", "med-recent"]
    print(f"{'frame':6s} " + " ".join(f"{v:>11s}" for v in variants) + f" {'literal':>9s}")
    agg = {v: [] for v in variants}
    lit = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(bl):
            print(f"[skip] {frame}")
            continue
        rm = io_utils.imread(os.path.join(bl, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(bl, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        lit_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        lit_d = io_utils.imread(os.path.join(bl, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        row = []
        for v in variants:
            c, d = aggregate(root, a.src_cam, list(range(a.n_frames)), rm, 10.0, 4.0, lvl,
                             rule=v, target=fi, recent=a.recent)
            sel = (c.any(2) | d.any())
            cc = np.array(lit_c, copy=True)
            dd = np.array(lit_d, copy=True)
            cc[rm & sel] = c[rm & sel]
            dd[rm & sel] = d[rm & sel]
            run = f"agg_{v}_{a.src_cam}{a.dst_cam}_{frame}"
            wd = io_utils.run_dir(run, create=True)
            for stage in ("20_warp", "30_class", "40_removal"):
                s = os.path.join(bl, stage)
                dst = os.path.join(wd, stage)
                os.makedirs(dst, exist_ok=True)
                for nm in os.listdir(s):
                    p = os.path.join(s, nm)
                    if os.path.isfile(p) and os.path.getsize(p) < 24 * 1024 * 1024:
                        shutil.copy2(p, os.path.join(dst, nm))
            sf = os.path.join(wd, "50_fill")
            os.makedirs(sf, exist_ok=True)
            io_utils.imwrite(os.path.join(sf, "filled_occlusion.png"), cc)
            io_utils.imwrite(os.path.join(sf, "filled_occlusion_depth.png"), dd)
            io_utils.imwrite(os.path.join(sf, "depth_pred.png"),
                             io_utils.imread(os.path.join(bl, "40_removal", "depth_pred.png"),
                                             gray=True))
            subprocess.run([PY, os.path.join(ROOT, "step6_render.py"), "--run", run,
                            "--src_cam", str(a.src_cam), "--dst_cam", str(a.dst_cam),
                            "--frame", frame], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=ROOT)
            fin = os.path.join(wd, "60_final", "final.png")
            if not os.path.isfile(fin):
                row.append(float("nan"))
                continue
            w = os.path.join(wd, "20_warp")
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            region = (dis | cr) & ~oofa
            val = metrics.psnr(io_utils.imread(fin), gt, mask=region)
            agg[v].append(val)
            row.append(val)
        lit.append(metrics.psnr(io_utils.imread(os.path.join(bl, "60_final", "final.png")),
                                gt, mask=region))
        print(f"{frame:6s} " + " ".join(f"{v:11.2f}" for v in row) + f" {lit[-1]:9.2f}")
    print(f"\n  论文原样  {float(np.mean(lit)):6.2f} dB")
    for v in variants:
        if agg[v]:
            print(f"  {v:11s} {float(np.nanmean(agg[v])):6.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
