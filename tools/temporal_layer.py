"""Layered composite: temporal background layer + plain foreground warp (proper z-buffer).

Warping the temporal background alone loses half the disocclusion pixels, because the true
background surface is FARTHER than the foreground depth the removed band sat at, so the
samples land elsewhere and the foreground object does not cover them.  The correct
construction is a two-layer composite in the VIRTUAL view:

    layer 0  the plain warp of the reference (foreground + the background that was visible)
    layer 1  the temporal background model warped with its own (true background) depth

and per destination pixel keep the NEARER sample (larger inverse depth).  Where the foreground
lands it wins (it was occluding in the reference too); where the disocclusion opens, only the
background layer has a sample and it fills the hole with REAL background content instead of a
flat extension.

    python tools/temporal_layer.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, warp as W


def build_bg(root, cam, frames, region, q=10.0, tol=4.0):
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
    H, Wd = region.shape
    bg_c = np.zeros((H, Wd, 3), np.uint8)
    bg_d = np.zeros(region.shape, np.uint8)
    bg_c[ys[ok], xs[ok]] = np.clip(np.rint(col[ok]), 0, 255).astype(np.uint8)
    bg_d[ys[ok], xs[ok]] = np.clip(np.rint(z_bg[ok]), 0, 255).astype(np.uint8)
    return bg_c, bg_d, n_s, ys, xs, ok


def layered_composite(cams, src, dst, ref_color, ref_depth, bg_c, bg_d,
                      bg_fill_color=None, bg_fill_depth=None):
    """Two-layer virtual view: reference warp (nearer wins) + background layer underneath."""
    fg = W.warp_view(cams, src, dst, ref_color, ref_depth)
    bg = W.warp_view(cams, src, dst, bg_c, bg_d)
    H, Wd = ref_depth.shape
    out_c = np.array(fg["warped_color"], copy=True)
    out_d = np.array(fg["warped_invdepth"], copy=True)     # larger = nearer
    fg_valid = ~fg["hole"]
    bg_valid = ~bg["hole"]
    # background wins only where the foreground has no sample, or is strictly farther
    take = bg_valid & (~fg_valid | (bg["warped_invdepth"] > out_d + 1e-6))
    take &= ~fg_valid | (out_d <= 0)
    out_c[take] = bg["warped_color"][take]
    out_d[take] = bg["warped_invdepth"][take]
    return out_c, out_d, dict(fg=fg, bg=bg, take=take)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--q", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=4.0)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    frames_t = list(range(a.n_frames))
    print(f"{'frame':6s} {'fg only':>8s} {'layer cov':>10s} {'layer PSNR':>11s} "
          f"{'shipped':>8s} {'hole px':>8s}")
    res = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            print(f"[skip] {frame}")
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
        occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        final = io_utils.imread(os.path.join(base, "60_final", "final.png"))
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"),
                              gray=True) > 0
        cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"),
                             gray=True) > 0
        reg = dis | cr
        bg_c, bg_d, n_s, ys, xs, ok = build_bg(root, a.src_cam, frames_t, rm, a.q, a.tol)
        # the background layer needs a depth everywhere it will be warped from, and the
        # reference background outside the removed band is already correct in ref_depth
        lay_c = np.array(ref["color"], copy=True)
        lay_d = np.array(ref["depth"], copy=True)
        sel_ok = np.zeros(rm.shape, bool)
        sel_ok[ys, xs] = ok
        lay_c[sel_ok] = bg_c[sel_ok]
        lay_d[sel_ok] = bg_d[sel_ok]
        out_c, out_d, st = layered_composite(cams, a.src_cam, a.dst_cam,
                                             ref["color"], ref["depth"], lay_c, lay_d)
        # score: only inside the disocclusion + cracks
        img = np.array(out_c, copy=True)
        val = metrics.psnr(img, gt, mask=reg)
        cov = int((~(st["fg"]["hole"])).sum())
        fg_only = metrics.psnr(st["fg"]["warped_color"], gt, mask=reg)
        shipped = metrics.psnr(final, gt, mask=reg)
        print(f"{frame:6s} {fg_only:8.2f} {cov:10d} {val:11.2f} {shipped:8.2f} "
              f"{int(reg.sum()):8d}")
        res.append((frame, fg_only, val, shipped))
    if res:
        m = lambda i: float(np.mean([r[i] for r in res]))
        print(f"\nMEAN  plain warp {m(1):.2f}  layered-temporal {m(2):.2f}  "
              f"shipped {m(3):.2f}   ->  temporal gain {m(2) - m(3):+.2f} dB over shipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
