"""Test the position-bias fix: the temporal depth clamp is forcing the WRONG depth.

Diagnosis (tools/depth_clean_check.py, BA54 f000):

    temporal model's own depth (covered pixels)  median 144   <- matches the reference
    reference depth at those same pixels         median 144   <- the true background surface
    the clamp used (stage 4's background level)  median  43..93
    after clamping                               median  43   <- 100+ units WRONG

Stage 4's background level is measured from the pixels AROUND the hole, which on this data is
not the level of the surface behind the hole, so using it as a hard clip destroys the temporal
model's depth.  Depth is geometry, so that misplacement is what shows up as the over-textured
"purple" blocks.

This A/B compares three occlusion layers, each composited through the run's own stage 6 path by
writing the layer into a scratch run and invoking step6_render.py:

    clamped   the current behaviour
    raw       the temporal model's own depth (no clamp)
    predict   the temporal colour but stage 4's predicted depth

    python tools/fix_position.py --src_cam 5 --dst_cam 4 --frames 0,5
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, temporal

PY = sys.executable
ROOT = r"D:\项目\空洞填补2"
SCRATCH = os.path.join(ROOT, "output", "_posfix")


def run_step(script, run, src, dst, frame, extra=()):
    cmd = [PY, os.path.join(ROOT, script), "--run", run, "--src_cam", str(src),
           "--dst_cam", str(dst), "--frame", frame] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--literal_run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,5")
    ap.add_argument("--n_frames", type=int, default=100)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    os.makedirs(SCRATCH, exist_ok=True)
    variants = ["clamped", "raw", "predict"]
    print(f"{'frame':6s} " + " ".join(f"{v:>9s}" for v in variants) +
          f" {'literal':>9s}  (空洞区 luma PSNR，经 step6 正规合成)")
    agg = {v: [] for v in variants}
    lit_v = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.literal_run, create=False),
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
        pred_d = io_utils.imread(os.path.join(bl, "40_removal", "depth_pred.png"), gray=True)
        m_clamped = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=lvl)
        m_raw = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=None)
        layers = {}
        layers["clamped"] = temporal.apply_to_occlusion_layer(lit_c, lit_d, m_clamped, rm)
        layers["raw"] = temporal.apply_to_occlusion_layer(lit_c, lit_d, m_raw, rm)
        c3, _, sel3 = temporal.apply_to_occlusion_layer(lit_c, lit_d, m_clamped, rm)
        d3 = np.array(lit_d, copy=True)
        d3[sel3] = pred_d[sel3]                     # temporal colour, predicted depth
        layers["predict"] = (c3, d3, sel3)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        row = []
        for v in variants:
            c, dd, sel = layers[v]
            run = f"posfix_{v}_{a.src_cam}{a.dst_cam}_{frame}"
            wd = os.path.join(io_utils.run_dir(run, create=True))
            # copy the literal run's inputs, then overwrite stage 5's outputs
            for stage in ("20_warp", "30_class", "40_removal"):
                s = os.path.join(bl, stage)
                ddst = os.path.join(wd, stage)
                os.makedirs(ddst, exist_ok=True)
                for n in os.listdir(s):
                    p = os.path.join(s, n)
                    if os.path.isfile(p) and os.path.getsize(p) < 24 * 1024 * 1024:
                        shutil.copy2(p, os.path.join(ddst, n))
            sf = os.path.join(wd, "50_fill")
            os.makedirs(sf, exist_ok=True)
            io_utils.imwrite(os.path.join(sf, "filled_occlusion.png"), c)
            io_utils.imwrite(os.path.join(sf, "filled_occlusion_depth.png"), dd)
            io_utils.imwrite(os.path.join(sf, "depth_pred.png"), pred_d)
            r = run_step("step6_render.py", run, a.src_cam, a.dst_cam, frame)
            fin = os.path.join(wd, "60_final", "final.png")
            if not os.path.isfile(fin):
                tail = (r.stderr or r.stdout or "").strip().splitlines()
                print(f"  [{frame} {v}] step6 failed: {tail[-1][:110] if tail else '?'}")
                row.append(float("nan"))
                continue
            img = io_utils.imread(fin)
            w = os.path.join(wd, "20_warp")
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            region = (dis | cr) & ~oofa
            val = metrics.psnr(img, gt, mask=region)
            agg[v].append(val)
            row.append(val)
        lit_v.append(metrics.psnr(io_utils.imread(os.path.join(bl, "60_final", "final.png")),
                                  gt, mask=region))
        print(f"{frame:6s} " + " ".join(f"{v:9.2f}" for v in row) + f" {lit_v[-1]:9.2f}")
    if lit_v:
        print(f"\n  论文原样            {float(np.mean(lit_v)):6.2f} dB")
        for v in variants:
            if agg[v]:
                print(f"  {v:18s} {float(np.nanmean(agg[v])):6.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
