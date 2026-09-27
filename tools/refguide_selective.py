"""Selective reference guidance: only where the temporal model has NO evidence.

Measured: reference-guided fill (search=0, depth_slack=8) gains +0.68 dB over the paper-literal
layer, but LOSES 0.67 dB when applied on top of the temporal model -- because it overwrites the
~35% of the band the temporal model already fills with real content, replacing an exact answer
with the nearest reference patch.

The temporal model covers ~30-40% of the band, so the remaining ~60-70% is exactly the part
that currently falls back to invention.  Applying reference guidance ONLY there should be
additive.

    python tools/refguide_selective.py --temporal 100 --src_cam 5 --dst_cam 4 --frames 0,1,2,3,4
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, refguide, temporal

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
                   capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    fin = os.path.join(wd, "60_final", "final.png")
    return io_utils.imread(fin) if os.path.isfile(fin) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2,3,4")
    ap.add_argument("--temporal", type=int, default=100)
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--search", type=int, default=0)
    ap.add_argument("--slack", type=float, default=8.0)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    print(f"{'frame':6s} {'literal':>8s} {'temporal':>9s} {'temp+ref(all)':>14s} "
          f"{'temp+ref(gap)':>14s} | {'temporal evid%':>14s} {'repl in gap%':>13s}")
    lit_v, tem_v, all_v, gap_v = [], [], [], []
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
        lit_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        lit_d = io_utils.imread(os.path.join(bl, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(bl, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        model = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=lvl)
        tem_c, tem_d, evid = temporal.apply_to_occlusion_layer(lit_c, lit_d, model, rm)
        rg = refguide.reference_guided_fill(
            tem_c, tem_d, ref["color"], pred_d, rm, cams, a.src_cam, a.dst_cam,
            search=a.search, patch=5, max_cost=48.0, depth_slack=a.slack)
        # selective: accept the reference value only where temporal had NO evidence
        rgap_c = np.array(tem_c, copy=True)
        rgap_d = np.array(tem_d, copy=True)
        gap = rm & ~evid
        accepted = rg["offset"].any(2)
        take = gap & accepted
        rgap_c[take] = rg["color"][take]
        rgap_d[take] = rg["depth"][take]
        w = os.path.join(bl, "20_warp")
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        row = []
        for tag, (c, d) in (("lit", (lit_c, lit_d)), ("tem", (tem_c, tem_d)),
                            ("all", (rg["color"], rg["depth"])), ("gap", (rgap_c, rgap_d))):
            im = compose(bl, f"rgs2_{tag}_{a.src_cam}{a.dst_cam}_{frame}", a.src_cam,
                         a.dst_cam, frame, c, d, pred_d)
            row.append(metrics.psnr(im, gt, mask=region) if im is not None else float("nan"))
        lit_v.append(row[0]); tem_v.append(row[1]); all_v.append(row[2]); gap_v.append(row[3])
        print(f"{frame:6s} {row[0]:8.2f} {row[1]:9.2f} {row[2]:14.2f} {row[3]:14.2f} | "
              f"{100 * evid[rm].mean():14.1f} "
              f"{100 * take.sum() / max(int(gap.sum()), 1):13.1f}")
    if lit_v:
        m = lambda v: float(np.nanmean(v))
        print(f"\n  论文原样            {m(lit_v):6.2f} dB")
        print(f"  +时序               {m(tem_v):6.2f} dB   ({m(tem_v) - m(lit_v):+.2f})")
        print(f"  时序 + 参考图(全部) {m(all_v):6.2f} dB   ({m(all_v) - m(tem_v):+.2f})")
        print(f"  时序 + 参考图(仅缺口) {m(gap_v):6.2f} dB   ({m(gap_v) - m(tem_v):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
