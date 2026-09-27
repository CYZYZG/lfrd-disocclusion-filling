"""Extend the temporal background spatially: the one avenue left open.

The temporal model is built per pixel from the sequence, so pixels that NEVER become
background in any frame get no evidence and fall back to the single-view prediction (on BA54
~60-70% of the removed band).  But the background is spatially smooth: a pixel with no
temporal evidence usually has a NEIGHBOUR that does, and their background surfaces are usually
continuous.  A spatial extension of the temporal model is therefore the natural complement to
the temporal model and mirrors what a patch-search fill does.

This measures how many uncovered pixels can be reached that way, and what it is worth:

  * "hole-fill": for each removed pixel without temporal evidence, take the nearest pixel WITH
    evidence within a radius, using its colour and depth
  * sweep the radius and report PSNR + texture retention against the shipped temporal result

    python tools/temporal_extend.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, temporal, warp as W


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def hf(img):
    return np.abs(cv2.Laplacian(luma(img).astype(np.float32), cv2.CV_32F, ksize=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--radii", default="0,1,2,3,5,8")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    radii = [int(x) for x in a.radii.split(",")]
    print(f"{'frame':6s} " + " ".join(f"r={r:<5d}" for r in radii) +
          f" | {'shipped':>8s} {'reach%':>7s}")
    agg = {r: [] for r in radii}
    base_v, reach_v = [], []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            continue
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        warp_c = io_utils.imread(os.path.join(base, "20_warp", "warped_color.png"))
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
        occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        # the SHIPPED result of this run: the temporal substitution is already baked into
        # 50_fill, so warp the occlusion layer as it stands -- that is what stage 6 does.
        r0 = W.warp_view(cams, a.src_cam, a.dst_cam, occ_c, occ_d)
        take0 = region & ~r0["hole"]
        shipped = np.array(warp_c, copy=True)
        shipped[take0] = r0["warped_color"][take0]
        p_ship = metrics.psnr(shipped, gt, mask=region)
        base_v.append(p_ship)

        rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(base, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        ref_lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        # the temporal model AS IT WAS APPLIED by stage 5 (same parameters)
        model = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=ref_lvl)
        # the occlusion layer stage 5 actually wrote already has the substitution applied, so
        # derive `valid` = where the temporal model contributed
        valid = model["valid"]
        # how much of the removed band is covered, and how far the uncovered pixels are from
        # the nearest covered one
        inv = (~valid & rm).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        d_unc = dist[rm & ~valid]
        reach = 100.0 * float((d_unc <= 5).mean()) if d_unc.size else 100.0
        reach_v.append(reach)

        fill_c = np.array(occ_c, copy=True)
        fill_d = np.array(occ_d, copy=True)
        row = []
        for rad in radii:
            c = np.array(occ_c, copy=True)
            dd = np.array(occ_d, copy=True)
            need = rm & ~valid
            if rad > 0 and need.any():
                # nearest covered pixel: label image = linear index of each covered pixel,
                # zeros elsewhere; distanceTransformWithLabels then reports, for every pixel,
                # the label of the nearest zero pixel -- i.e. the nearest covered pixel.
                label_img = np.zeros(valid.shape, np.int32)
                sy, sx = np.nonzero(valid)
                label_img[sy, sx] = (sy * valid.shape[1] + sx + 1).astype(np.int32)
                dist2, lbl = cv2.distanceTransformWithLabels(
                    (label_img == 0).astype(np.uint8), cv2.DIST_L2, 3,
                    labelType=cv2.DIST_LABEL_CCOMP)
                src_lin = lbl.astype(np.int64) - 1
                syy, sxx = np.divmod(src_lin, valid.shape[1])
                within = cv2.dilate(valid.astype(np.uint8),
                                    cv2.getStructuringElement(
                                        cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))) > 0
                newly = need & within
                if newly.any():
                    c[newly] = occ_c[syy[newly], sxx[newly]]
                    dd[newly] = occ_d[syy[newly], sxx[newly]]
            r = W.warp_view(cams, a.src_cam, a.dst_cam, c, dd)
            take = region & ~r["hole"]
            img = np.array(warp_c, copy=True)
            img[take] = r["warped_color"][take]
            v = metrics.psnr(img, gt, mask=region)
            row.append(v)
            agg[rad].append(v)
        print(f"{frame:6s} " + " ".join(f"{v:7.2f}" for v in row) +
              f" | {p_ship:8.2f} {reach:6.1f}%")
    if base_v:
        mb = float(np.mean(base_v))
        print(f"\n  时序基线（无空间扩散）  {mb:6.2f} dB    "
              f"未被时序覆盖的移除像素中 {np.mean(reach_v):.1f}% 在 5 px 内有覆盖")
        for rad in radii:
            if agg[rad]:
                mv = float(np.mean(agg[rad]))
                print(f"  半径 {rad:<3d}               {mv:6.2f} dB   ({mv - mb:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
