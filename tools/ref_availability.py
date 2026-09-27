"""Is the real texture available in the reference view, and are we throwing it away?

The A-oracle in tools/texture_oracle.py used the GROUND-TRUTH virtual depth to back-project
into the reference.  This one uses the geometry the PIPELINE actually predicts -- the
occlusion layer's own depth -- so it answers a very different question:

  for every disocclusion pixel, where does the pipeline's predicted depth point in the
  reference image, and does that location hold background or foreground?

If it mostly points at BACKGROUND, the real texture IS in the reference and our synthesis is
discarding it (a fixable matching/selection problem).  If it points at FOREGROUND or outside
the frame, the content genuinely is not there and only inpainting can help.

    python tools/ref_availability.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"{'frame':6s} {'hole px':>8s} {'ref-hit':>8s} {'of hit: BG':>10s} {'FG':>7s} "
          f"{'outside':>8s} | {'PSNR ours':>9s} {'PSNR ref-paste':>14s}")
    agg = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            continue
        ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.src_cam, frame)
        gtv = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)
        gt = gtv["color"]
        ours = io_utils.imread(os.path.join(base, "60_final", "final.png"))
        occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        fg = io_utils.imread(os.path.join(base, "30_class", "fg_mask.png"),
                             gray=True) > 0
        # the reference background separator used by the pipeline
        sep = float(np.percentile(ref["depth"][ref["depth"] > 0], 60))
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"),
                              gray=True) > 0
        oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"),
                               gray=True) > 0
        region = dis & ~oofa
        ys, xs = np.nonzero(region)

        # reproduce stage 6: warp the occlusion layer with the pipeline's own depth and take
        # the value that landed there.  For pixels the layer did NOT cover we cannot say where
        # the pipeline pointed, so they are counted separately.
        from lfrd import warp as W
        occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
        res = W.warp_view(cams, a.src_cam, a.dst_cam, occ_c, occ_d)
        covered = ~res["hole"][ys, xs]
        n_cov = int(covered.sum())

        # Where does a hole pixel's own predicted depth point in the reference?
        # Use the composite depth of the virtual view (what stage 6 wrote).
        comp = np.load(os.path.join(base, "60_final", "final_depth.npy")).astype(np.float64)
        z = comp[ys, xs]
        ut, vt, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), z,
                                       cams[a.dst_cam], cams[a.src_cam])
        H, W = gt.shape[:2]
        ur = np.rint(ut).astype(np.int64)
        vr = np.rint(vt).astype(np.int64)
        inb = ok & (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)
        n_all = ys.size
        hit = np.zeros(region.shape, bool)
        n_hit = int(inb.sum())
        if n_hit:
            hit[ys[inb], xs[inb]] = True
        # foreground test at the reference location
        is_fg = np.zeros(n_all, bool)
        if n_hit:
            ii = np.nonzero(inb)[0]
            is_fg[ii] = fg[vr[ii], ur[ii]] | (ref["depth"][vr[ii], ur[ii]] > sep)
        n_bg = int((inb & ~is_fg).sum())
        n_fg = int((inb & is_fg).sum())
        p_paste = float("nan")
        if n_bg > 50:
            ii = np.nonzero(inb & ~is_fg)[0]
            paste = np.array(ours, copy=True)
            paste[ys[ii], xs[ii]] = ref["color"][vr[ii], ur[ii]]
            p_paste = metrics.psnr(paste, gt, mask=region)
        p_ours = metrics.psnr(ours, gt, mask=region)
        ref_hit = 100.0 * n_hit / max(1, n_all)
        bgf = 100.0 * n_bg / max(1, n_all)
        fgf = 100.0 * n_fg / max(1, n_all)
        ouf = 100.0 * (n_all - n_hit) / max(1, n_all)
        print(f"{frame:6s} {int(region.sum()):8d} {ref_hit:7.1f}% {bgf:9.1f}% {fgf:6.1f}% "
              f"{ouf:7.1f}% | {p_ours:9.2f} {p_paste:14.2f}   (layer covered {n_cov})")
        agg.append((ref_hit, bgf, fgf, ouf, p_ours, p_paste, n_cov, int(region.sum())))
    if agg:
        m = lambda i: float(np.nanmean([r[i] for r in agg]))
        print(f"\nMEAN  预测深度在参考图中命中 {m(0):.1f}% 的空洞像素，其中背景 "
              f"{m(1):.1f}%、前景 {m(2):.1f}%；完全落空（出画/无效）{m(3):.1f}%")
        print(f"      PSNR ours {m(4):.2f} -> 把命中处的参考像素贴回去 {m(5):.2f}")
        print(f"      遮挡层实际覆盖 {m(6):.0f} / {m(7):.0f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
