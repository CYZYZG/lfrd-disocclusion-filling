"""Feather the boundary between the temporal background and the single-view prediction.

Observation from the galleries: the temporal background brings REAL texture, but its evidence
region has a hard edge -- a pixel either has temporal samples or it does not -- and the result
reads as patchy / blocky where the two contents meet (visible in gallery_zoom_f005/f008).

Fix under test: build a soft weight from the temporal evidence mask (blur it a few px) and
blend the temporal layer with the single-view occlusion layer across that band, instead of
cutting between them.

    python tools/feather_probe.py --src_cam 5 --dst_cam 4 --frames 0,3,5,8
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, temporal, warp as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--literal_run", default="ba54_seq")
    ap.add_argument("--temporal_run", default="ba54_latest")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,3,5,8")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--sigmas", default="0,1,2,3,5")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    sigmas = [float(x) for x in a.sigmas.split(",")]
    print(f"{'frame':6s} {'temporal px':>11s} " +
          " ".join(f"s={s:<5.0f}" for s in sigmas) + f" | {'shipped':>8s}")
    agg = {s: [] for s in sigmas}
    ship = []
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
        warp_c = io_utils.imread(os.path.join(wdir, "warped_color.png"))
        dis = io_utils.imread(os.path.join(wdir, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(wdir, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(wdir, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        rm = io_utils.imread(os.path.join(bl, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        lit_c = io_utils.imread(os.path.join(bl, "50_fill", "filled_occlusion.png"))
        lit_d = io_utils.imread(os.path.join(bl, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        tem_c = io_utils.imread(os.path.join(bt, "50_fill", "filled_occlusion.png"))
        tem_d = io_utils.imread(os.path.join(bt, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        # where did the temporal model contribute?
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(bl, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        ref_lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        model = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=ref_lvl)
        valid = model["valid"].astype(np.float32)
        n_val = int(model["valid"].sum())
        row = []
        for s in sigmas:
            if s <= 0:
                wgt = valid
            else:
                k = int(2 * round(3 * s) + 1)
                wgt = cv2.GaussianBlur(valid, (k, k), s)
            wgt = np.clip(wgt, 0.0, 1.0)
            # blend colour and depth between the literal and temporal occlusion layers
            blend_c = (lit_c.astype(np.float32) * (1 - wgt[..., None]) +
                       tem_c.astype(np.float32) * wgt[..., None])
            blend_d = (lit_d.astype(np.float32) * (1 - wgt) + tem_d.astype(np.float32) * wgt)
            c = np.clip(np.rint(blend_c), 0, 255).astype(np.uint8)
            dd = np.clip(np.rint(blend_d), 0, 255).astype(np.uint8)
            r = W.warp_view(cams, a.src_cam, a.dst_cam, c, dd)
            take = region & ~r["hole"]
            img = np.array(warp_c, copy=True)
            img[take] = r["warped_color"][take]
            v = metrics.psnr(img, gt, mask=region)
            row.append(v)
            agg[s].append(v)
        shipped = metrics.psnr(io_utils.imread(os.path.join(bt, "60_final", "final.png")),
                               gt, mask=region)
        ship.append(shipped)
        print(f"{frame:6s} {n_val:11d} " + " ".join(f"{v:7.2f}" for v in row) +
              f" | {shipped:8.2f}")
    if ship:
        mb = float(np.mean(ship))
        print(f"\n  当前 shipped（硬边）      {mb:6.2f} dB")
        for s in sigmas:
            if agg[s]:
                mv = float(np.mean(agg[s]))
                print(f"  羽化 sigma={s:<4.0f}        {mv:6.2f} dB   ({mv - mb:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
