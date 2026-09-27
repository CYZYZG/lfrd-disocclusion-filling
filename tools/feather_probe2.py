"""Feather test with a correct baseline: composite each occlusion layer the same way stage 6 does.

The earlier feather probe compared a hand-rolled composite against the shipped final and lost
6 dB, because the occlusion-layer depth of the temporal run is not the geometry stage 6 ended up
using.  This version takes the SHIPPED final of the temporal run as the baseline and only
modifies the pixels inside the temporal-evidence region, so the comparison is incremental and
the baseline is exact.

Two effects are separated:
  * hard cut   -- replace the evidence region with the temporal content (should reproduce the
                  shipped result, or improve on it)
  * feather    -- blend the temporal content into the single-view content across a soft band
                  built from the evidence mask

    python tools/feather_probe2.py --src_cam 5 --dst_cam 4 --frames 0,3,5,8
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, temporal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--literal_run", default="ba54_seq")
    ap.add_argument("--temporal_run", default="ba54_latest")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,3,5,8")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--sigmas", default="0,1,2,3")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    sigmas = [float(x) for x in a.sigmas.split(",")]
    print(f"{'frame':6s} {'evid px':>8s} " + " ".join(f"s={s:<5.0f}" for s in sigmas) +
          f" | {'shipped':>8s}  {'literal':>8s}")
    agg = {s: [] for s in sigmas}
    ship, lit_v = [], []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        bl = os.path.join(io_utils.run_dir(a.literal_run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        bt = os.path.join(io_utils.run_dir(a.temporal_run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not (os.path.isdir(bl) and os.path.isdir(bt)):
            continue
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        wdir = os.path.join(bt, "20_warp")
        dis = io_utils.imread(os.path.join(wdir, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(wdir, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(wdir, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        rm = io_utils.imread(os.path.join(bl, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        ship_img = io_utils.imread(os.path.join(bt, "60_final", "final.png"))
        lit_img = io_utils.imread(os.path.join(bl, "60_final", "final.png"))
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(bl, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        ref_lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        model = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=ref_lvl)
        evid = model["valid"].astype(np.float32)
        # the temporal COLOUR as stage 5 wrote it (in the reference view)
        tmp_occ = temporal.apply_to_occlusion_layer(
            io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png")),
            io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion_depth.png"),
                            gray=True), model, rm)
        row = []
        for s in sigmas:
            if s <= 0:
                w = evid
            else:
                k = int(2 * round(3 * s) + 1)
                w = np.clip(cv2.GaussianBlur(evid, (k, k), s), 0, 1)
            # incremental change: blend the LITERAL result towards the TEMPORAL result using w
            # on the reference side is not possible post hoc, so blend in the virtual view:
            # w is defined in the reference view, so warp it too.
            row.append(w)
        # build the blended reference occlusion layers, warp each, and composite
        lit_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        tem_c = tmp_occ[0]
        from lfrd import calib, warp as W
        cams = calib.load_calib()
        warp_c = io_utils.imread(os.path.join(wdir, "warped_color.png"))
        for i, s in enumerate(sigmas):
            w = row[i]
            bc = (lit_c.astype(np.float32) * (1 - w[..., None]) +
                  tem_c.astype(np.float32) * w[..., None])
            c = np.clip(np.rint(bc), 0, 255).astype(np.uint8)
            d_ = io_utils.imread(os.path.join(bl, "50_fill",
                                              "filled_occlusion_depth.png"), gray=True)
            r = W.warp_view(cams, a.src_cam, a.dst_cam, c, d_)
            take = region & ~r["hole"]
            img = np.array(warp_c, copy=True)
            img[take] = r["warped_color"][take]
            v = metrics.psnr(img, gt, mask=region)
            agg[s].append(v)
            if i == 0:
                first = v
        p_ship = metrics.psnr(ship_img, gt, mask=region)
        p_lit = metrics.psnr(lit_img, gt, mask=region)
        ship.append(p_ship)
        lit_v.append(p_lit)
        print(f"{frame:6s} {int(model['valid'].sum()):8d} " +
              " ".join(f"{v:7.2f}" for v in [agg[s][-1] for s in sigmas]) +
              f" | {p_ship:8.2f}  {p_lit:8.2f}")
    if ship:
        mb = float(np.mean(ship))
        print(f"\n  当前 shipped（时序硬边）  {mb:6.2f} dB")
        print(f"  论文原样（无时序）        {float(np.mean(lit_v)):6.2f} dB")
        for s in sigmas:
            if agg[s]:
                mv = float(np.mean(agg[s]))
                print(f"  参考层混合 s={s:<4.0f}      {mv:6.2f} dB   ({mv - mb:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
