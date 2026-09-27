"""Measure the reference-guided occlusion-layer fill before wiring it in.

Runs the real stage 5 to get the current occlusion layer, then applies
lfrd.refguide.reference_guided_fill and composites both through the real stage 6, so the
comparison uses the pipeline's own geometry and OOFA handling (hand-built composites kept
disagreeing with stage 6 by several dB, see tools/stage6_accounting.py).

    python tools/refguide_probe.py --src_cam 5 --dst_cam 4 --frames 0,1,2 [--temporal 100]
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, refguide, temporal

PY = sys.executable
ROOT = r"D:\项目\空洞填补2"


def prepare_run(base_src, run, src, dst, frame, layer_c, layer_d, pred_d):
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
    io_utils.imwrite(os.path.join(sf, "filled_occlusion.png"), layer_c)
    io_utils.imwrite(os.path.join(sf, "filled_occlusion_depth.png"), layer_d)
    io_utils.imwrite(os.path.join(sf, "depth_pred.png"), pred_d)
    r = subprocess.run([PY, os.path.join(ROOT, "step6_render.py"), "--run", run,
                        "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT)
    fin = os.path.join(wd, "60_final", "final.png")
    return fin if os.path.isfile(fin) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--temporal", type=int, default=0, help="temporal_frames for the base layer")
    ap.add_argument("--search", type=int, default=6)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--max_cost", type=float, default=48.0)
    ap.add_argument("--depth_slack", type=float, default=8.0)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    from lfrd import calib
    cams = calib.load_calib()
    print(f"{'frame':6s} {'base':>8s} {'+refguide':>10s} {'delta':>7s} {'repl%':>6s} "
          f"{'cost':>6s} | {'covered-gain':>13s}")
    base_v, rg_v = [], []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.base_run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(bl):
            print(f"[skip] {frame}")
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        rm = io_utils.imread(os.path.join(bl, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        pred_d = io_utils.imread(os.path.join(bl, "40_removal", "depth_pred.png"), gray=True)
        lay_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        lay_d = io_utils.imread(os.path.join(bl, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        if a.temporal:
            meta = None
            try:
                meta = io_utils.load_npz(os.path.join(bl, "40_removal", "removal_meta.npz"))
            except Exception:                                          # noqa: BLE001
                pass
            lvl = meta.get("bg_level") if isinstance(meta, dict) else None
            model = temporal.build_temporal_background(
                root, a.src_cam, list(range(a.temporal)), rm, q=10.0, tol=4.0,
                ref_level=lvl)
            lay_c, lay_d, _ = temporal.apply_to_occlusion_layer(lay_c, lay_d, model, rm)
        rg = refguide.reference_guided_fill(
            lay_c, lay_d, ref["color"], pred_d, rm, cams, a.src_cam, a.dst_cam,
            search=a.search, patch=a.patch, max_cost=a.max_cost,
            depth_slack=a.depth_slack, verbose=True)
        recs = []
        for tag, (c, d) in (("base", (lay_c, lay_d)),
                            ("+refguide", (rg["color"], rg["depth"]))):
            fin = prepare_run(bl, f"rg_{tag}_{a.src_cam}{a.dst_cam}_{frame}",
                              a.src_cam, a.dst_cam, frame, c, d, pred_d)
            if fin is None:
                recs.append((tag, float("nan"), None))
                continue
            img = io_utils.imread(fin)
            w = os.path.join(bl, "20_warp")
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            region = (dis | cr) & ~oofa
            recs.append((tag, metrics.psnr(img, gt, mask=region), img))
        b = recs[0][1]
        r = recs[1][1]
        base_v.append(b)
        rg_v.append(r)
        print(f"{frame:6s} {b:8.2f} {r:10.2f} {r - b:+7.2f} "
              f"{100 * rg['accepted_frac']:6.1f} {rg['mean_cost']:6.1f}")
    if base_v:
        print(f"\n  base        {float(np.nanmean(base_v)):6.2f} dB")
        print(f"  +refguide   {float(np.nanmean(rg_v)):6.2f} dB   "
              f"({float(np.nanmean(rg_v)) - float(np.nanmean(base_v)):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
