"""Headroom analysis, done correctly.

The paper's central idea is: remove the local foreground in the REFERENCE view, invent the
occluded background there (colour + depth), then forward-warp that invented layer into the
virtual view to fill the hole.  So the question "how much better could this get?" splits into

  (a) how good is the invented OCCLUSION LAYER (reference-view colour + depth of the removed
      region) -- this is step 5, the paper's contribution, and
  (b) how much of the hole is even reachable by that mechanism.

This script builds an ORACLE occlusion layer by taking the ground truth of the virtual camera
and forward-warping it back into the removed region of the reference view (z-buffered), then
runs our own step-6 compositing on it.  Comparing that against our actual result bounds what
any improvement of steps 5.1/5.2 can buy.

    python tools/oracle_ceiling.py --run q_67 --src_cam 6 --dst_cam 7 --frames 0,1,2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, warp as warp_mod


def gray(a):
    a = np.asarray(a, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def psnr(gt, img, mask):
    d = (gray(gt) - gray(img)) ** 2
    m = np.asarray(mask, bool)
    if m.ndim == 3:
        m = m[..., 0]
    d = d[m]
    return float(10 * np.log10(255.0 ** 2 / d.mean())) if d.size else float("nan")


def backwarp_gt_to_reference(cams, dst, src, gt_color, gt_depthP, rem_mask, chunk=4096):
    """z-buffered forward warp of the virtual-view ground truth into the reference frame.

    Only writes pixels of `rem_mask` (the region our method removed), so the oracle gives
    exactly "the content that should have been invented there".
    """
    H, W = gt_depthP.shape
    ys, xs = np.nonzero(np.ones_like(gt_depthP, bool))
    z = calib.depth_from_P(gt_depthP.astype(np.float64))
    z_inv = np.where(z > 0, 1.0 / np.maximum(z, 1e-6), 0.0).astype(np.float32)
    occ_c = np.zeros((H, W, 3), np.float64)
    occ_d = np.zeros((H, W), np.float64)
    best = np.full((H, W), -1.0, np.float32)
    col = gt_color.reshape(-1, 3)
    zflat = gt_depthP.reshape(-1)
    for i in range(0, ys.size, chunk):
        sy = ys[i:i + chunk].astype(np.float64)
        sx = xs[i:i + chunk].astype(np.float64)
        zi = z[ys[i:i + chunk], xs[i:i + chunk]]
        ut, vt, ok = calib.project_pts(sx, sy, zi, cams[dst], cams[src])
        u = np.rint(ut).astype(np.int64)
        v = np.rint(vt).astype(np.int64)
        m = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zi > 0)
        if not m.any():
            continue
        yy, xx = v[m], u[m]
        zz = z_inv[ys[i:i + chunk], xs[i:i + chunk]][m]
        keep = zz > best[yy, xx]
        yy, xx, zz = yy[keep], xx[keep], zz[keep]
        idx = (i + np.nonzero(m)[0])[keep]
        best[yy, xx] = zz
        occ_c[yy, xx] = col[idx]
        occ_d[yy, xx] = zflat[idx]
    out_mask = rem_mask & (best > 0)
    return occ_c.astype(np.uint8), occ_d.astype(np.uint8), out_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"{'frame':6s} {'region px':>9s} {'warp':>7s} {'OURS':>7s} {'ORACLE-layer':>13s} "
          f"{'headroom':>9s} {'oracle covers':>13s}")
    agg = {"warp": [], "ours": [], "orc": [], "cov": []}
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        root = io_utils.run_dir(a.run, create=False)
        fd = os.path.join(root, f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(fd, "20_warp")
        if not os.path.isdir(w):
            continue
        ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.src_cam, frame)
        gt_view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)
        gt = gt_view["color"]
        ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        crack = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        region = (dis | crack) & ~oofa
        rm = io_utils.imread(os.path.join(root, "40_removal", "removed_mask.png"),
                             gray=True) > 0

        # oracle occlusion layer: reference colour/depth with the removed region replaced by
        # the ground truth warped back from the virtual camera
        occ_c = np.array(ref["color"])
        occ_d = np.array(ref["depth"])
        oc, od, om = backwarp_gt_to_reference(cams, a.dst_cam, a.src_cam, gt,
                                              gt_view["depth"], rm)
        occ_c[om] = oc[om]
        occ_d[om] = od[om]
        ww = warp_mod.warp_view(cams, a.src_cam, a.dst_cam, occ_c, occ_d)
        comp = np.array(ours, copy=True)
        m_ok = region & ~ww["hole"]
        comp[m_ok] = ww["warped_color"][m_ok]
        orc = psnr(gt, comp, region)
        cov = 100.0 * m_ok.sum() / max(1, region.sum())
        print(f"{frame:6s} {int(region.sum()):9d} {psnr(gt, warp, region):7.2f} "
              f"{psnr(gt, ours, region):7.2f} {orc:13.2f} {orc - psnr(gt, ours, region):+9.2f} "
              f"{cov:12.1f}%")
        agg["warp"].append(psnr(gt, warp, region))
        agg["ours"].append(psnr(gt, ours, region))
        agg["orc"].append(orc)
        agg["cov"].append(cov)
    if agg["ours"]:
        m = lambda k: float(np.mean(agg[k]))
        print(f"\nMEAN  warp {m('warp'):.2f}  OURS {m('ours'):.2f}  "
              f"ORACLE-occlusion-layer {m('orc'):.2f}  "
              f"(headroom {m('orc') - m('ours'):+.2f} dB, oracle reaches "
              f"{m('cov'):.1f}% of the hole)")
        print("=> every dB of this headroom is what a better step 5.1 (depth prediction) and")
        print("   step 5.2 (occlusion-layer texture) could buy.  Anything beyond it needs")
        print("   information that a single reference view does not contain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
