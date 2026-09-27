"""Temporal background model for the removed (occluded) region.

Paper IV / V: the occluded background that disocclusions expose is *not* recoverable from a
single view, but in a video it usually is -- the foreground moves and the same background
shows up in other frames of the SAME (static) camera.  For MSR Ballet the camera is fixed:
the depth of a pixel is either the background surface or the moving dancer, so a per-pixel
temporal MINIMUM of the inverse depth is a background estimate and the samples that reach it
carry the true background colour.

    z_bg(p)   = percentile_t( P_t(p), q )            (q = 10 by default)
    S(p)      = { t : P_t(p) <= z_bg(p) + tol }
    colour(p) = median_{t in S(p)} I_t(p)

This module builds that model for the region stage 4 removed, and can score it end to end by
running stage 5 (inpaint only the few pixels that never become visible) + stage 6.

    python tools/temporal_bg.py --run ba54_seq --src_cam 5 --dst_cam 4 --frame f000 --measure
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, inpaint as I, io_utils, metrics, warp as W

CACHE = os.path.join(ROOT, "output", "_temporal")


def _load_stack(root, cam, frames, kind):
    """(n, H, W[, 3]) stack of one camera's frames."""
    out = []
    for f in frames:
        name = (f"color-cam{cam}-f{f:03d}.jpg" if kind == "color"
                else f"depth-cam{cam}-f{f:03d}.png")
        p = os.path.join(root, f"cam{cam}", name)
        if not os.path.isfile(p):
            continue
        if kind == "color":
            out.append(io_utils.imread(p))
        else:
            out.append(io_utils.imread(p, gray=True))
    return np.stack(out) if out else None


def build_background(root, cam, frames, region, q=10.0, tol=4.0, chunk=8):
    """Return (bg_color, bg_depth, n_samples) for the pixels of `region`.

    Colors/depths are only computed on `region` (the removed band is ~50k px, so a full-frame
    stack would be 100x more memory than needed).
    """
    ys, xs = np.nonzero(region)
    n = len(frames)
    P = np.empty((n, ys.size), np.int16)
    COL = np.empty((n, ys.size, 3), np.uint8)
    for i, f in enumerate(frames):
        P[i] = io_utils.imread(os.path.join(root, f"cam{cam}",
                                            f"depth-cam{cam}-f{f:03d}.png"),
                               gray=True)[ys, xs]
        COL[i] = io_utils.imread(os.path.join(root, f"cam{cam}",
                                              f"color-cam{cam}-f{f:03d}.jpg"))[ys, xs]
    z_bg = np.percentile(P, q, axis=0)
    near = (P <= (z_bg[None, :] + tol)) & (P > 0)
    n_s = near.sum(0)
    col = np.zeros((ys.size, 3), np.float32)
    for i in range(n):
        sel = near[i]
        if sel.any():
            col[sel] += COL[i][sel].astype(np.float32)
    ok = n_s > 0
    col[ok] /= n_s[ok, None]
    bg_c = np.zeros((region.shape[0], region.shape[1], 3), np.uint8)
    bg_d = np.zeros(region.shape, np.uint8)
    bg_c[ys[ok], xs[ok]] = np.clip(np.rint(col[ok]), 0, 255).astype(np.uint8)
    bg_d[ys[ok], xs[ok]] = np.clip(np.rint(z_bg[ok]), 0, 255).astype(np.uint8)
    return bg_c, bg_d, n_s.reshape(-1), ys, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--q", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--min_samples", type=int, default=1)
    ap.add_argument("--measure", action="store_true", help="run step6 and score")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    frame = a.frame
    base = os.path.join(io_utils.run_dir(a.run, create=False),
                        f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
    rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"),
                         gray=True) > 0
    frames = list(range(a.n_frames))
    print(f"building temporal background for {int(rm.sum())} px from {len(frames)} frames "
          f"(q={a.q}, tol={a.tol}) ...")
    bg_c, bg_d, n_s, ys, xs = build_background(root, a.src_cam, frames, rm, a.q, a.tol)
    print(f"  pixels with >=1 background sample: {int((n_s >= a.min_samples).sum())} "
          f"({100.0 * (n_s >= a.min_samples).mean():.1f}%)")
    print(f"  samples per pixel: median {np.median(n_s):.0f}, p10 "
          f"{np.percentile(n_s, 10):.0f}, max {n_s.max()}")
    # comparison with the current prediction: how different are the colours?
    occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
    pred_c = io_utils.imread(os.path.join(base, "40_removal", "removed_color.png"))
    covered = n_s >= a.min_samples
    diff = np.abs(bg_c[rm].astype(int) - occ_c[rm].astype(int)).mean()
    print(f"  mean |temporal - current occlusion layer| colour diff: {diff:.1f}")
    # build the modified occlusion layer: temporal background where we have samples, the
    # existing inpainting elsewhere
    mod_c = np.array(occ_c, copy=True)
    mod_d = np.array(io_utils.imread(os.path.join(base, "50_fill",
                                                  "filled_occlusion_depth.png"),
                                     gray=True), copy=True)
    sel = np.zeros(rm.shape, bool)
    sel[ys[covered], xs[covered]] = True
    mod_c[sel] = bg_c[sel]
    mod_d[sel] = bg_d[sel]
    os.makedirs(CACHE, exist_ok=True)
    io_utils.imwrite(os.path.join(CACHE, f"temporal_cam{a.src_cam}_{frame}.png"), mod_c)
    io_utils.imwrite(os.path.join(CACHE, f"temporal_depth_cam{a.src_cam}_{frame}.png"), mod_d)
    np.save(os.path.join(CACHE, f"nsamples_cam{a.src_cam}_{frame}.npy"), n_s)

    if a.measure:
        cams = calib.load_calib()
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"),
                              gray=True) > 0
        cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"),
                             gray=True) > 0
        warp_c = io_utils.imread(os.path.join(base, "20_warp", "warped_color.png"))
        final = io_utils.imread(os.path.join(base, "60_final", "final.png"))
        reg = dis | cr
        cur_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        # hybrid: the temporal background COLOUR (real content) placed with the pipeline's
        # predicted depth.  The temporal depth is the true background surface, which is FARTHER
        # than the foreground depth the removed band sits at, so warping it moves the samples
        # out of the target pixels (coverage halved).  Keeping the current depth makes the two
        # comparable and isolates "better content" from "different geometry".
        hyb_c = np.array(occ_c, copy=True)
        hyb_c[sel] = bg_c[sel]
        for name, (oc, od) in (("current", (occ_c, cur_d)),
                               ("temporal-depth", (mod_c, mod_d)),
                               ("temporal-colour", (hyb_c, cur_d))):
            r = W.warp_view(cams, a.src_cam, a.dst_cam, oc, od)
            take = reg & ~r["hole"]
            img = np.array(warp_c, copy=True)
            img[take] = r["warped_color"][take]
            print(f"  {name:16s}: covers {int(take.sum())} of {int(reg.sum())} fill px "
                  f"-> PSNR {metrics.psnr(img, gt, mask=take):.2f} dB")
        print(f"  (the shipped pipeline scores "
              f"{metrics.psnr(final, gt, mask=reg):.2f} dB on the same region)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
