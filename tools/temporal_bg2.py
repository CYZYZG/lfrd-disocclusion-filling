"""Temporal background model, constrained by stage 4's measured background level.

`tools/temporal_bg.py` takes a per-pixel temporal percentile of the inverse depth.  That is
only a background estimate for pixels that DO become background at some frame; a pixel that
stays occluded in every frame keeps a foreground value, and mixing such pixels into the
background model is what made the first attempt inconsistent (temporal depth median 160 vs
the 95 that stage 4 measured as the revealed background level).

This version therefore:
  * computes, per removed-run, the same background level stage 4 measured (from the run's
    background-classified edge pixels) as a reference;
  * clips the per-pixel temporal percentile with that level and counts how many pixels had to
    be clipped (those never became background);
  * exposes the sample count so the caller can require evidence before using the model.

    python tools/temporal_bg2.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, warp as W


def temporal_model(root, cam, frames, region, q=10.0, tol=4.0, ref_level=None):
    """(bg_color, bg_depth, n_samples, clipped) on the pixels of `region`."""
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
    z = np.percentile(P, q, axis=0).astype(np.float32)
    clipped = np.zeros(ys.size, bool)
    if ref_level is not None:
        lvl = np.asarray(ref_level, np.float32)[ys, xs]
        ok = np.isfinite(lvl) & (lvl > 0)
        over = ok & (z > lvl + 3.0)
        z[over] = lvl[over]
        clipped = over
    near = (P <= (z[None, :] + tol)) & (P > 0)
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
    bg_d[ys[ok], xs[ok]] = np.clip(np.rint(z[ok]), 0, 255).astype(np.uint8)
    return bg_c, bg_d, n_s, clipped


def run_cell(run, src, dst, frame, n_frames=100, q=10.0, tol=4.0, use_level=True,
             verbose=True):
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    base = os.path.join(io_utils.run_dir(run, create=False), f"cam{src}-cam{dst}-{frame}")
    rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"),
                         gray=True) > 0
    bg_level = None
    if use_level:
        meta = io_utils.load_npz(os.path.join(base, "40_removal", "removal_meta.npz"))
        bg_level = meta["bg_level"] if "bg_level" in meta else None
    bg_c, bg_d, n_s, clipped = temporal_model(root, src, list(range(n_frames)), rm,
                                             q, tol, bg_level)
    ys, xs = np.nonzero(rm)
    okv = n_s >= 1
    occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
    occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                         "filled_occlusion_depth.png"), gray=True)
    lay_c = np.array(occ_c, copy=True)
    lay_d = np.array(occ_d, copy=True)
    sel = np.zeros(rm.shape, bool)
    sel[ys[okv], xs[okv]] = True
    lay_c[sel] = bg_c[sel]
    lay_d[sel] = bg_d[sel]
    gt = io_utils.load_view(root, dst, frame)["color"]
    dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"), gray=True) > 0
    cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"), gray=True) > 0
    warp_c = io_utils.imread(os.path.join(base, "20_warp", "warped_color.png"))
    final = io_utils.imread(os.path.join(base, "60_final", "final.png"))
    reg = dis | cr
    out = {}
    for name, (oc, od) in (("current", (occ_c, occ_d)), ("temporal", (lay_c, lay_d))):
        r = W.warp_view(cams, src, dst, oc, od)
        take = reg & ~r["hole"]
        img = np.array(warp_c, copy=True)
        img[take] = r["warped_color"][take]
        out[name] = dict(psnr=metrics.psnr(img, gt, mask=take), cover=int(take.sum()),
                         psnr_all=metrics.psnr(img, gt, mask=reg))
    out["shipped"] = metrics.psnr(final, gt, mask=reg)
    out["n_samples_med"] = float(np.median(n_s))
    out["clipped_frac"] = float(clipped.mean())
    out["sel_px"] = int(sel.sum())
    if verbose:
        print(f"{frame}: samples med {out['n_samples_med']:.0f} "
              f"clipped {100 * out['clipped_frac']:.1f}%  | "
              f"current {out['current']['psnr']:.2f} (cov {out['current']['cover']})  "
              f"temporal {out['temporal']['psnr']:.2f} (cov {out['temporal']['cover']})  "
              f"shipped {out['shipped']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--q", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--no_level", action="store_true")
    a = ap.parse_args()
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        rows.append(run_cell(a.run, a.src_cam, a.dst_cam, frame, a.n_frames, a.q, a.tol,
                             use_level=not a.no_level))
    if rows:
        mc = float(np.mean([r["current"]["psnr"] for r in rows]))
        cc = float(np.mean([r["current"]["cover"] for r in rows]))
        mt = float(np.mean([r["temporal"]["psnr"] for r in rows]))
        ct = float(np.mean([r["temporal"]["cover"] for r in rows]))
        ms = float(np.mean([r["shipped"] for r in rows]))
        clip = 100 * float(np.mean([r["clipped_frac"] for r in rows]))
        print(f"\nMEAN  current {mc:.2f} (cover {cc:.0f} px)   temporal {mt:.2f} "
              f"(cover {ct:.0f} px)   shipped {ms:.2f}")
        print(f"      clipped (never became background): {clip:.1f}% of removed px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
