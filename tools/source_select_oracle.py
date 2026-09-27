"""Choosing among three candidate sources: where is the remaining headroom?

Three sources exist for a disocclusion pixel:
  1. the single-view occlusion layer (paper-literal pipeline)
  2. the temporal background model (real content from other frames where visible)
  3. the reference image at the back-projected position (real texture, where that position is
     reference background)

Measured on BA54, source 3 alone lifts the paper-literal result +1.3 dB but *hurts* the
temporal result (-0.7 dB), so the sources are complementary and a per-pixel choice is what is
missing.  This measures the selection ORACLE, i.e. the ceiling any GT-free selector could
reach, which bounds how much work a selector is worth.

    python tools/source_select_oracle.py --run ba54_seq --temporal_run ba54_temporal `
        --src_cam 5 --dst_cam 4 --frames 0-3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics


def candidates(run, src, dst, frame, cams, bg_tol=0.0):
    base = os.path.join(io_utils.run_dir(run, create=False), f"cam{src}-cam{dst}-{frame}")
    ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, src, frame)
    dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"), gray=True) > 0
    cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"), gray=True) > 0
    region = (dis | cr) & ~oofa
    ours = io_utils.imread(os.path.join(base, "60_final", "final.png"))
    comp = np.load(os.path.join(base, "60_final", "final_depth.npy")).astype(np.float64)
    fg = io_utils.imread(os.path.join(base, "30_class", "fg_mask.png"), gray=True) > 0
    dref = ref["depth"]
    sep = float(np.percentile(dref[dref > 0], 60)) + bg_tol
    ys, xs = np.nonzero(region)
    z = comp[ys, xs]
    with np.errstate(invalid="ignore"):
        ut, vt, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), z,
                                       cams[dst], cams[src])
    H, W = ref["color"].shape[:2]
    bad = ~np.isfinite(ut) | ~np.isfinite(vt)
    ur = np.where(bad, -1, np.rint(np.nan_to_num(ut))).astype(np.int64)
    vr = np.where(bad, -1, np.rint(np.nan_to_num(vt))).astype(np.int64)
    inb = ok & ~bad & (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H) & (z > 0)
    is_bg = np.zeros(ys.size, bool)
    ii = np.nonzero(inb)[0]
    if ii.size:
        is_bg[ii] = (~fg[vr[ii], ur[ii]]) & (dref[vr[ii], ur[ii]] <= sep) \
            & (dref[vr[ii], ur[ii]] > 0)
    return region, ys, xs, ours, ref, ur, vr, is_bg, ii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="paper-literal run")
    ap.add_argument("--temporal_run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2,3")
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"{'frame':6s} {'hole':>7s} {'paper':>7s} {'+temporal':>10s} {'+ref':>7s} "
          f"{'oracle3':>8s} {'temporal':>9s} {'tv':>6s} | best-source histogram")
    agg = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        (region, ys, xs, ours, ref, ur, vr, is_bg,
         ii) = candidates(a.run, a.src_cam, a.dst_cam, frame, cams)
        if not region.any():
            continue
        tp = os.path.join(io_utils.run_dir(a.temporal_run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}", "60_final", "final.png")
        temporal = io_utils.imread(tp) if os.path.isfile(tp) else ours
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]

        m = region.copy()
        # candidate images: A paper-literal, B temporal, C paper-literal + reference paste
        A = np.array(ours, copy=True)
        B = np.array(temporal, copy=True)
        C = np.array(ours, copy=True)
        jj = np.nonzero(is_bg)[0]
        if jj.size:
            C[ys[jj], xs[jj]] = ref["color"][vr[jj], ur[jj]]
        pA = metrics.psnr(A, gt, mask=m)
        pB = metrics.psnr(B, gt, mask=m)
        pC = metrics.psnr(C, gt, mask=m)
        # oracle over the three
        err = np.stack([np.abs(A.astype(np.float32) - gt.astype(np.float32)).mean(2),
                        np.abs(B.astype(np.float32) - gt.astype(np.float32)).mean(2),
                        np.abs(C.astype(np.float32) - gt.astype(np.float32)).mean(2)], 0)
        pick = np.argmin(err, 0)
        best = np.where((pick == 0)[..., None], A,
                        np.where((pick == 1)[..., None], B, C))
        pO = metrics.psnr(best, gt, mask=m)
        hist = [int(((pick == k) & m).sum()) for k in range(3)]
        print(f"{frame:6s} {int(m.sum()):7d} {pA:7.2f} {pB:10.2f} {pC:7.2f} {pO:8.2f} "
              f"{pB - pA:+9.2f} {pO - pB:+6.2f} | A {100 * hist[0] / m.sum():.0f}% "
              f"B {100 * hist[1] / m.sum():.0f}% C {100 * hist[2] / m.sum():.0f}%")
        agg.append((int(m.sum()), pA, pB, pC, pO, hist))
    if agg:
        s = [sum(r[5]) for r in agg]
        tot = sum(s)
        m = lambda i: float(np.nanmean([r[i] for r in agg]))
        print(f"\nMEAN  paper-literal {m(1):.2f}   +temporal {m(2):.2f}   "
              f"+reference-paste {m(3):.2f}   ORACLE(3-way) {m(4):.2f}")
        print(f"      最优来源占比：paper {100 * s[0] / tot:.0f}%  temporal "
              f"{100 * s[1] / tot:.0f}%  ref-paste {100 * s[2] / tot:.0f}%")
        print(f"      => 一个完美的无 GT 选择器最多再拿 {m(4) - m(2):+.2f} dB"
              f"（相对当前最好的 时序 结果）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
