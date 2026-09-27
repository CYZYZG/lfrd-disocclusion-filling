"""Spatially varying photometric correction: estimate the drift field WITHOUT ground truth.

The residual map inside the hole is not flat (f000: block values from +1 to -17, spread 15.1
across blocks, while the global correction only removes 6.89).  So a single per-channel
offset+gain leaves a structured error behind.

The correction field has to be estimated without ground truth.  The seam is where the answer is
known: at the boundary the filled content should join the surrounding valid content.  So:

    offset0 = the per-pixel difference between the fill and the valid content at the seam
    offset  = that seam offset diffused inwards over the hole (the fill's drift is a smooth
              function of position, so its boundary value is a good estimate for the interior)
    result  = fill - offset

Four ways of building the field are compared against the shipped global correction:

    global      the current behaviour (one offset+gain per channel)
    diffuse     seam offset, diffused inwards (Laplace inpainting of the offset field)
    poly1/poly2 a least-squares plane / quadratic fitted to the seam offset, evaluated in the hole
    percomp     one offset per connected component of the filled region

    python tools/photo_spatial.py --run best_final --src_cam 5 --dst_cam 4 --frames 0,5
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics


def seam_band(hole, valid, ring):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    inner = hole & (cv2.erode(hole.astype(np.uint8), k) == 0)
    outer = valid & (cv2.dilate(hole.astype(np.uint8), k) > 0)
    return inner, outer


def global_correct(final, hole, valid, ring=3, clip=25.0, contrast=True):
    f = final.astype(np.float32)
    out = np.array(final, copy=True)
    inner, outer = seam_band(hole, valid, ring)
    if inner.sum() < 20 or outer.sum() < 20:
        return np.array(final, copy=True)
    for c in range(3):
        hh, vv = f[..., c][inner], f[..., c][outer]
        gain = 1.0
        if contrast and hh.std() > 1e-3 and vv.std() > 1e-3:
            gain = float(np.clip(vv.std() / hh.std(), 0.8, 1.25))
        shift = float(np.clip(vv.mean() - gain * hh.mean(), -clip, clip))
        out[..., c][hole] = np.clip(gain * f[..., c][hole] + shift, 0, 255).astype(np.uint8)
    return out


def diffuse_field(field, hole, known, iters=400, lam=0.9):
    """Laplace diffusion: interpolate `field` from `known` pixels into `hole`."""
    f = field.astype(np.float32)
    m = known.astype(np.float32)
    for _ in range(iters):
        avg = cv2.blur(f, (3, 3))
        wavg = cv2.blur(m, (3, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            cand = avg / np.maximum(wavg, 1e-6)
        upd = hole & (wavg > 1e-6)
        f[upd] = (1 - lam) * f[upd] + lam * cand[upd]
        m[upd] = 1.0
    return f


def spatial_correct(final, hole, valid, mode="diffuse", ring=3, clip=25.0, order=1,
                    iters=400):
    f = final.astype(np.float32)
    out = np.array(final, copy=True)
    inner, outer = seam_band(hole, valid, ring)
    if inner.sum() < 20 or outer.sum() < 20:
        return np.array(final, copy=True)
    H, W = hole.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for c in range(3):
        base = f[..., c]
        if mode == "diffuse":
            # seam samples: (fill - valid_neighbourhood) at the seam
            known = np.zeros(hole.shape, bool)
            field = np.zeros(hole.shape, np.float32)
            # for each inner pixel take the mean difference to nearby outer pixels
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
            cnt = cv2.blur(outer.astype(np.float32), (2 * ring + 1, 2 * ring + 1))
            mean_out = cv2.blur(np.where(outer, base, 0.0).astype(np.float32),
                                (2 * ring + 1, 2 * ring + 1))
            with np.errstate(invalid="ignore", divide="ignore"):
                nb = mean_out / np.maximum(cnt, 1e-6)
            sel = inner & (cnt > 1e-6)
            field[sel] = (base - nb)[sel]
            known[sel] = True
            fld = diffuse_field(field, hole, known, iters=iters)
        else:
            # polynomial fitted to the seam offset
            sel = inner
            z = np.zeros_like(base)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
            cnt = cv2.blur(outer.astype(np.float32), (2 * ring + 1, 2 * ring + 1))
            mean_out = cv2.blur(np.where(outer, base, 0.0).astype(np.float32),
                                (2 * ring + 1, 2 * ring + 1))
            with np.errstate(invalid="ignore", divide="ignore"):
                nb = mean_out / np.maximum(cnt, 1e-6)
            sel = inner & (cnt > 1e-6)
            z[sel] = (base - nb)[sel]
            ys, xs = np.nonzero(sel)
            if ys.size < 10:
                return np.array(final, copy=True)
            cols = [np.ones_like(xs, np.float64)]
            xa = (xs - W / 2.0) / (W / 2.0)
            ya = (ys - H / 2.0) / (H / 2.0)
            cols += [xa, ya]
            if order >= 2:
                cols += [xa * xa, ya * ya, xa * ya]
            A = np.stack(cols, 1)
            coef, *_ = np.linalg.lstsq(A, z[sel].astype(np.float64), rcond=None)
            hy, hx = np.nonzero(hole)
            xh = (hx - W / 2.0) / (W / 2.0)
            yh = (hy - H / 2.0) / (H / 2.0)
            hcols = [np.ones_like(xh, np.float64), xh, yh]
            if order >= 2:
                hcols += [xh * xh, yh * yh, xh * yh]
            Ah = np.stack(hcols, 1)
            fld = np.zeros(hole.shape, np.float32)
            fld[hy, hx] = (Ah @ coef).astype(np.float32)
        fld = np.clip(fld, -clip, clip)
        out[..., c][hole] = np.clip(base[hole] - fld[hole], 0, 255).astype(np.uint8)
    return out


def percomp_correct(final, hole, valid, ring=3, clip=25.0):
    f = final.astype(np.float32)
    out = np.array(final, copy=True)
    n, lab = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        comp = lab == i
        if comp.sum() < 50:
            continue
        inner, outer = seam_band(comp, valid | (hole & ~comp), ring)
        if inner.sum() < 10 or outer.sum() < 10:
            continue
        for c in range(3):
            off = float(np.clip(f[..., c][outer].mean() - f[..., c][inner].mean(),
                                -clip, clip))
            out[..., c][comp] = np.clip(f[..., c][comp] + off, 0, 255).astype(np.uint8)
    return out


def hybrid_correct(final, hole, valid, ring=3, clip=25.0, order=1, iters=400, strength=0.5,
                   mode="diffuse"):
    """Global mean/contrast match FIRST, then remove the residual drift it leaves behind.

    The spatial variants alone lose to the global one because the global step matches the mean of
    the WHOLE filled region to the seam, while a local estimate only matches locally and leaves a
    global bias uncorrected.  So: apply the global correction, re-estimate the seam offset on the
    corrected image (which is now small), and remove `strength` of it.
    """
    g = global_correct(final, hole, valid, ring=ring, clip=clip)
    f = g.astype(np.float32)
    out = np.array(g, copy=True)
    inner, outer = seam_band(hole, valid, ring)
    if inner.sum() < 20 or outer.sum() < 20 or strength <= 0:
        return g
    H, W = hole.shape
    ks = 2 * ring + 1
    cnt = cv2.blur(outer.astype(np.float32), (ks, ks))
    for c in range(3):
        base = f[..., c]
        mean_out = cv2.blur(np.where(outer, base, 0.0).astype(np.float32), (ks, ks))
        with np.errstate(invalid="ignore", divide="ignore"):
            nb = mean_out / np.maximum(cnt, 1e-6)
        sel = inner & (cnt > 1e-6)
        if sel.sum() < 20:
            return g
        if mode == "diffuse":
            field = np.zeros(hole.shape, np.float32)
            known = np.zeros(hole.shape, bool)
            field[sel] = (base - nb)[sel]
            known[sel] = True
            fld = diffuse_field(field, hole, known, iters=iters)
        else:
            ys, xs = np.nonzero(sel)
            z = (base - nb)[sel].astype(np.float64)
            xa = (xs - W / 2.0) / (W / 2.0)
            ya = (ys - H / 2.0) / (H / 2.0)
            cols = [np.ones_like(xa), xa, ya]
            if order >= 2:
                cols += [xa * xa, ya * ya, xa * ya]
            A = np.stack(cols, 1)
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            hy, hx = np.nonzero(hole)
            xh = (hx - W / 2.0) / (W / 2.0)
            yh = (hy - H / 2.0) / (H / 2.0)
            hcols = [np.ones_like(xh), xh, yh]
            if order >= 2:
                hcols += [xh * xh, yh * yh, xh * yh]
            fld = np.zeros(hole.shape, np.float32)
            fld[hy, hx] = (np.stack(hcols, 1) @ coef).astype(np.float32)
        fld = np.clip(fld, -clip, clip) * float(strength)
        out[..., c][hole] = np.clip(base[hole] - fld[hole], 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="best_final")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0")
    a = ap.parse_args()
    strengths = [float(x) for x in a.strengths.split(",")]
    modes = (["none", "global"] + [f"hyb_d{int(s * 100)}" for s in strengths]
             + [f"hyb_p{int(s * 100)}" for s in strengths])
    agg = {m: [] for m in modes}
    print(f"{'frame':6s} " + " ".join(f"{m:>8s}" for m in modes))
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(d):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        w = os.path.join(d, "20_warp")
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        hole = (dis | cr) & ~oofa
        valid = ~(dis | cr | oofa)
        ppp = os.path.join(d, "60_final", "final_pre_post.png")
        src_img = io_utils.imread(ppp) if os.path.isfile(ppp) else \
            io_utils.imread(os.path.join(d, "60_final", "final.png"))
        imgs = {"none": src_img, "global": global_correct(src_img, hole, valid)}
        for s in strengths:
            if s <= 0:
                imgs[f"hyb_d{int(s * 100)}"] = imgs["global"]
                imgs[f"hyb_p{int(s * 100)}"] = imgs["global"]
                continue
            imgs[f"hyb_d{int(s * 100)}"] = hybrid_correct(
                src_img, hole, valid, mode="diffuse", strength=s, iters=a.iters)
            imgs[f"hyb_p{int(s * 100)}"] = hybrid_correct(
                src_img, hole, valid, mode="poly", order=1, strength=s)
        row = []
        for m in modes:
            v = metrics.psnr(imgs[m], gt, mask=hole)
            agg[m].append(v)
            row.append(v)
        print(f"{frame:6s} " + " ".join(f"{v:8.3f}" for v in row))
    if agg["none"]:
        m = lambda k: float(np.nanmean(agg[k]))
        base = m("none")
        print()
        for k in modes:
            if agg[k]:
                print(f"  {k:10s} {m(k):7.3f} dB   ({m(k) - base:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
