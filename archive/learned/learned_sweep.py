"""Does the learned (sparse) fill stack with the temporal model, and with which parameters?

First result: a per-channel PCA dictionary fitted on the scene's own background patches
(30000 patches -> 48 atoms, 99.8% explained variance, 0.2 s to fit) fills the removed band to
+0.51 dB over the existing occlusion layer, through the real stage 6.

Now the questions that decide whether it is worth wiring in:
  * does it ADD to the temporal model, or overlap with it like reference_guided did?
  * which patch size / atom count / damping is best?

    python tools/learned_sweep.py --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, learned, metrics, temporal

PY = sys.executable
ROOT = r"D:\项目\空洞填补2"


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
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--temporal", type=int, default=100)
    ap.add_argument("--grid", default="9:48,9:16,9:96,7:48,11:48,9:48@4")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    combos = []
    for spec in a.grid.split(","):
        if "@" in spec:
            p, rest = spec.split("@")
            patch, atoms = p.split(":")
            combos.append((int(patch), int(atoms), float(rest)))
        else:
            patch, atoms = spec.split(":")
            combos.append((int(patch), int(atoms), 1.0))
    print(f"{'frame':6s} {'patch:atoms:lam':>16s} {'base':>7s} {'leanred':>8s} {'delta':>7s} "
          f"{'+temporal':>10s} {'tem+learned':>12s} {'delta':>7s}")
    sums = {}
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(bl):
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        rm = io_utils.imread(os.path.join(bl, "40_removal", "removed_mask.png"), gray=True) > 0
        fg = io_utils.imread(os.path.join(bl, "30_class", "fg_mask.png"), gray=True) > 0
        pred_d = io_utils.imread(os.path.join(bl, "40_removal", "depth_pred.png"), gray=True)
        lay_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        lay_d = io_utils.imread(os.path.join(bl, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(bl, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        model = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.temporal)), rm, q=10.0, tol=4.0, ref_level=lvl)
        tem_c, tem_d, _ = temporal.apply_to_occlusion_layer(lay_c, lay_d, model, rm)
        dref = ref["depth"].astype(np.float64)
        nz = dref[dref > 0]
        allowed = (~fg) & (dref > 0) & (dref <= float(np.percentile(nz, 40)))
        w = os.path.join(bl, "20_warp")
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        # one composite per arm
        def score(tag, c, d):
            im = compose(bl, f"ls_{tag}_{a.src_cam}{a.dst_cam}_{frame}", a.src_cam,
                         a.dst_cam, frame, c, d, pred_d)
            return metrics.psnr(im, gt, mask=region) if im is not None else float("nan")
        p_lit = score("lit", lay_c, lay_d)
        p_tem = score("tem", tem_c, tem_d)
        for (patch, atoms, lam) in combos:
            key = f"{patch}:{atoms}:{lam:g}"
            M, C, _ = learned.collect_patches(ref["color"], allowed, patch=patch, stride=3,
                                              max_patches=30000)
            if M.shape[1] < 200:
                continue
            dic = learned.fit_dictionary(M, n_atoms=atoms)
            lc, st1 = learned.dictionary_fill(lay_c, rm, ~rm, dic, patch=patch,
                                              n_atoms=atoms, lam=lam, max_iters=300000)
            tc, st2 = learned.dictionary_fill(tem_c, rm, ~rm, dic, patch=patch,
                                              n_atoms=atoms, lam=lam, max_iters=300000)
            pl = score(f"l{patch}_{atoms}", lc, lay_d)
            pt = score(f"tl{patch}_{atoms}", tc, tem_d)
            print(f"{frame:6s} {key:>16s} {p_lit:7.2f} {pl:8.2f} {pl - p_lit:+7.2f} "
                  f"{p_tem:10.2f} {pt:12.2f} {pt - p_tem:+7.2f}")
            sums.setdefault(key, []).append((pl - p_lit, pt - p_tem))
    print()
    for key, v in sums.items():
        arr = np.array(v)
        print(f"  {key:16s} 论文原样 {arr[:, 0].mean():+.3f}   +时序 {arr[:, 1].mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
