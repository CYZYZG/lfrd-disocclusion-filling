"""Depth-consistent reference re-selection: the refinement the measurements point to.

What the diagnostics established (BA54 cam5->cam4, frames 0-2):

  * every disocclusion pixel back-projects into the reference frame, but only ~31% onto
    reference BACKGROUND                (tools/ref_availability.py)
  * pasting the reference colour at those positions lifts the paper-literal result
    +1.29 dB and lifts the texture-band HF energy from 0.78 to 1.09 of ground truth
    (tools/ref_reselect.py) -- i.e. real texture is available and our synthesis discards it
  * it HURTS the temporal result (-0.74 dB) because the temporal content is better there
  * a per-pixel oracle over {paper, temporal, reference-paste} reaches +0.97 dB above the
    temporal result, but every confidence rule tried so far is NEGATIVE
    (tools/selector_search.py)

The naive paste failed twice for the same reason: the reference pixel it takes is at the
WRONG DEPTH.  A pixel can project onto a reference-background location that belongs to a
different surface (a nearer background layer), and then the colour is simply misplaced.  So
this version adds the consistency test the earlier attempts lacked:

    accept the reference sample only when |P_ref(u,v) - P_target_predicted| <= tol

i.e. the reference sample must be the SAME SURFACE the pipeline says is behind the hole.  Then
sweep tol and the temporal-evidence guard.

    python tools/ref_depth_consistent.py --run ba54_seq --temporal_run ba54_temporal `
        --src_cam 5 --dst_cam 4 --frames 0,1,2,3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics


def load(run, src, dst, frame, cams, n_frames=100, temporal_run=None):
    base = os.path.join(io_utils.run_dir(run, create=False), f"cam{src}-cam{dst}-{frame}")
    ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, src, frame)
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
    dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"), gray=True) > 0
    cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"), gray=True) > 0
    region = (dis | cr) & ~oofa
    A = io_utils.imread(os.path.join(base, "60_final", "final.png"))
    B = A
    if temporal_run:
        tp = os.path.join(io_utils.run_dir(temporal_run, create=False),
                          f"cam{src}-cam{dst}-{frame}", "60_final", "final.png")
        if os.path.isfile(tp):
            B = io_utils.imread(tp)
    # stage 6 composite depth = our predicted background depth in the hole
    pred = np.load(os.path.join(base, "60_final", "final_depth.npy")).astype(np.float64)
    fg = io_utils.imread(os.path.join(base, "30_class", "fg_mask.png"), gray=True) > 0
    # the occlusion layer's own predicted depth is the surface we claim is behind the hole
    occ_d = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion_depth.png"),
                            gray=True)
    ys, xs = np.nonzero(region)
    z = pred[ys, xs]
    with np.errstate(invalid="ignore"):
        ut, vt, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), z,
                                       cams[dst], cams[src])
    H, W = gt.shape[:2]
    bad = ~np.isfinite(ut) | ~np.isfinite(vt)
    ur = np.where(bad, -1, np.rint(np.nan_to_num(ut))).astype(np.int64)
    vr = np.where(bad, -1, np.rint(np.nan_to_num(vt))).astype(np.int64)
    inb = ok & ~bad & (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H) & (z > 0)
    is_bg = np.zeros(ys.size, bool)
    ii = np.nonzero(inb)[0]
    if ii.size:
        is_bg[ii] = ~fg[vr[ii], ur[ii]]
    # the surface depth the reference pixel actually holds
    pref = np.zeros(ys.size, np.float32)
    if ii.size:
        pref[ii] = ref["depth"][vr[ii], ur[ii]]
    return dict(base=base, ref=ref, gt=gt, region=region, A=A, B=B, ys=ys, xs=xs,
                ur=ur, vr=vr, inb=inb, is_bg=is_bg, pref=pref, z=z, pred=pred,
                occ_d=occ_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--temporal_run", default=None)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2,3")
    ap.add_argument("--n_frames", type=int, default=100)
    a = ap.parse_args()
    cams = calib.load_calib()
    tols = [0.0, 2.0, 4.0, 8.0, 16.0]
    print("在时序结果的基础上，用【深度一致性】筛选参考图重选")
    print(f"{'frame':6s} " + " ".join(f"tol={t:<4.0f}" for t in tols) +
          f" | {'base(temporal)':>14s} {'cover%':>7s}")
    agg = {t: [] for t in tols}
    base_v = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = load(a.run, a.src_cam, a.dst_cam, frame, cams, a.n_frames, a.temporal_run)
        reg, gt = d["region"], d["gt"]
        if not reg.any():
            continue
        B, A = d["B"], d["A"]
        # depth consistency: reference surface depth must match the predicted hole depth
        rel = np.zeros(d["ys"].size)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(d["pref"] - d["z"]) / np.maximum(d["z"], 1e-6)
        row = []
        for t in tols:
            ok_sel = d["inb"] & d["is_bg"] & (rel * 100.0 <= t)
            img = np.array(B, copy=True)
            jj = np.nonzero(ok_sel)[0]
            if jj.size:
                img[d["ys"][jj], d["xs"][jj]] = d["ref"]["color"][d["vr"][jj], d["ur"][jj]]
            v = metrics.psnr(img, gt, mask=reg)
            row.append(v)
            agg[t].append(v)
        base_v.append(metrics.psnr(B, gt, mask=reg))
        cover = 100.0 * (d["inb"] & d["is_bg"]).sum() / max(1, int(reg.sum()))
        print(f"{frame:6s} " + " ".join(f"{v:7.2f}" for v in row) +
              f" | {base_v[-1]:14.2f} {cover:6.1f}%")
    print()
    mb = float(np.mean(base_v))
    for t in tols:
        if agg[t]:
            mv = float(np.mean(agg[t]))
            print(f"  tol={t:<5.0f}   {mv:6.2f} dB   ({mv - mb:+.2f} vs 时序基线)")
    print(f"\n  时序基线（不重选）  {mb:6.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
