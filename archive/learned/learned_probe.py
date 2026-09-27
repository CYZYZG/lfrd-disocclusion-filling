"""Validate the learned (sparse-representation) occlusion-layer fill, through the real stage 6.

    python tools/learned_probe.py --src_cam 5 --dst_cam 4 --frames 0 --atoms 48
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

from lfrd import io_utils, learned, metrics

PY = sys.executable
ROOT = r"D:\项目\空洞填补2"
SCRATCH = os.path.join(ROOT, "output", "_learned")


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
    ap.add_argument("--frames", default="0")
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--atoms", type=int, default=48)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_patches", type=int, default=30000)
    ap.add_argument("--max_iters", type=int, default=200000)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    os.makedirs(SCRATCH, exist_ok=True)
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(bl):
            print(f"[skip] {frame}")
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        rm = io_utils.imread(os.path.join(bl, "40_removal", "removed_mask.png"), gray=True) > 0
        fg = io_utils.imread(os.path.join(bl, "30_class", "fg_mask.png"), gray=True) > 0
        pred_d = io_utils.imread(os.path.join(bl, "40_removal", "depth_pred.png"), gray=True)
        lay_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        lay_d = io_utils.imread(os.path.join(bl, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)

        # ---- train the dictionary on this scene's own background patches ---------- #
        dref = ref["depth"].astype(np.float64)
        nz = dref[dref > 0]
        lvl = float(np.percentile(nz, 40))
        allowed = (~fg) & (dref > 0) & (dref <= lvl)
        t0 = time.time()
        M, C, _ = learned.collect_patches(ref["color"], allowed, patch=a.patch,
                                          stride=a.stride, max_patches=a.max_patches)
        if M.shape[1] < 200:
            print(f"{frame}: only {M.shape[1]} background patches, skipped")
            continue
        dic = learned.fit_dictionary(M, n_atoms=a.atoms)
        t_train = time.time() - t0
        ev = float(dic["eigvals"].sum() / max(np.var(M, axis=1).sum(), 1e-9))
        print(f"{frame}: dictionary from {M.shape[1]} background patches x {C} channels -> "
              f"{dic['basis'].shape[2]} atoms in {t_train:.1f}s (top-{dic['basis'].shape[2]} "
              f"explained variance {ev:.3f})")

        # ---- fill the removed band with the learned model ------------------------- #
        t0 = time.time()
        new_c, st = learned.dictionary_fill(lay_c, rm, ~rm, dic, patch=a.patch,
                                            n_atoms=a.atoms, lam=a.lam,
                                            max_iters=a.max_iters, verbose=False)
        t_fill = time.time() - t0
        print(f"   fill: {st['filled_px']} px, {st['iters']} iters, {t_fill:.1f}s "
              f"(leftover {st['leftover']})")

        # composite both through the real stage 6
        w = os.path.join(bl, "20_warp")
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        recs = {}
        for tag, (c, d) in (("base", (lay_c, lay_d)), ("learned", (new_c, lay_d))):
            im = compose(bl, f"learned_{tag}_{a.src_cam}{a.dst_cam}_{frame}", a.src_cam,
                         a.dst_cam, frame, c, d, pred_d)
            recs[tag] = metrics.psnr(im, gt, mask=region) if im is not None else float("nan")
        print(f"   hole PSNR: base {recs['base']:.2f} -> learned {recs['learned']:.2f} "
              f"({recs['learned'] - recs['base']:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
