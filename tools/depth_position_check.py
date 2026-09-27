"""Where does the temporal model's POSITION bias come from?

Depth is geometry: for the BA54 pair a one-unit change in the 8-bit inverse depth moves a
sample by roughly a pixel.  So if the temporal model's background depth disagrees with what the
pipeline predicts, the real texture it brings lands in the wrong place -- which is exactly the
"over-textured" purple blocks in tools/artifact_map.py.

This measures, on the removed band of one frame:

  A  the depth the temporal model produced           z_temporal
  B  stage 4's per-row predicted background depth    z_pred      (what stage 6 used)
  C  the reference depth at pixels that are BACKGROUND in the target frame
                                                     z_ref_bg    (ground truth for a static camera)

and converts the disagreement into the horizontal pixel displacement it implies, per camera
pair, so it is clear how damaging it is.

    python tools/depth_position_check.py --run ba54_seq --src_cam 5 --dst_cam 4 --frame f000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, temporal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="f000,f003,f005,f008")
    ap.add_argument("--n_frames", type=int, default=100)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    src, dst = a.src_cam, a.dst_cam
    Ks, Kd = cams[src], cams[dst]
    print("相机对 %d->%d" % (src, dst))
    print(f"{'frame':6s} {'rm px':>7s} | {'z_pred':>7s} {'z_temporal':>10s} "
          f"{'z_ref_bg':>9s} | {'|dt|':>5s} {'|dp|':>5s} | {'px err(t)':>9s} {'px err(p)':>9s}")
    for frame in a.frames.split(","):
        d = os.path.join(io_utils.run_dir(a.run, create=False), f"cam{src}-cam{dst}-{frame}")
        if not os.path.isdir(d):
            continue
        ref = io_utils.load_view(root, src, frame)
        rm = io_utils.imread(os.path.join(d, "40_removal", "removed_mask.png"), gray=True) > 0
        fg = io_utils.imread(os.path.join(d, "30_class", "fg_mask.png"), gray=True) > 0
        pred = io_utils.imread(os.path.join(d, "40_removal", "depth_pred.png"),
                               gray=True).astype(np.float64)
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(d, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        model = temporal.build_temporal_background(
            root, src, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=lvl)
        zt = model["bg_depth"].astype(np.float64)[rm]
        zp = pred[rm]
        # ground truth for a static camera: the reference depth at pixels that are background
        bgmask = rm & ~fg & (ref["depth"] > 0)
        zr = ref["depth"].astype(np.float64)[bgmask]
        # the same pixels' temporal / predicted values, for a like-for-like comparison
        zt_b = model["bg_depth"].astype(np.float64)[bgmask]
        zp_b = pred[bgmask]
        # displacement implied by a depth difference, per unit of 8-bit inverse depth
        # dx/dP from the projection: x = f*X/Z + cx with Z = 1/((P/255)(1/MinZ-1/MaxZ)+1/MaxZ)
        MinZ, MaxZ = 42.0, 130.0
        P = np.linspace(20, 240, 400)
        Z = 1.0 / ((P / 255.0) * (1.0 / MinZ - 1.0 / MaxZ) + 1.0 / MaxZ)
        # depth-derived displacement between the two views for the same surface point
        X = np.zeros_like(Z)                       # assume the optical axis point (worst case)
        # empirical: use the pair's known disparity gradient instead of a projection derivation
        base = abs(Kd["t"][0] - Ks["t"][0])
        f_px = Ks["K"][0, 0]
        # disparity for the same 3D point between two views with baseline b: d = f*b/Z
        disp = f_px * base / Z
        dPdx = np.gradient(disp, P)
        conv = float(np.median(np.abs(dPdx)))
        print(f"{frame:6s} {int(rm.sum()):7d} | {np.median(zp):7.0f}   {np.median(zt):8.0f}  "
              f"{np.median(zr) if zr.size else float('nan'):8.0f}  | "
              f"{np.median(np.abs(zt_b - zr)) if zr.size else float('nan'):5.0f} "
              f"{np.median(np.abs(zp_b - zr)) if zr.size else float('nan'):5.0f} | "
              f"{np.median(np.abs(zt_b - zr)) * conv if zr.size else float('nan'):9.1f} "
              f"{np.median(np.abs(zp_b - zr)) * conv if zr.size else float('nan'):9.1f}")
    print(f"\n换算：1 个 8-bit 逆深度单位 ≈ {conv:.2f} px 横向位移（该相机对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
