"""Compare the two filling ROUTES on the same warp: synthetic-view search vs occlusion-layer warp.

The sibling reproduction (dibr/inpaint.py) fills a disocclusion like this:

    template = the SYNTHETIC view's already-filled content around the target
    search   = a 69x69 window in the REFERENCE image centred on the target's BACKWARD WARP
    source   = reference background only, adaptive 9->7->5->3, source must lie in ONE depth layer

Our pipeline instead predicts the removed band in the REFERENCE view (paper III-C/D), warps that
occlusion layer into the virtual view (III-E), and inpaints whatever it does not cover.

On smooth background and near the hole boundary the sibling is 3.29 / 3.07 dB ahead, and the
question is whether that comes from the direct route.  This implements the sibling's route in our
own code (so both use our calibration, our masks and our stage 6 bookkeeping) and scores them on
the same problem.

    python tools/route_compare.py --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics

DIAMOND7 = ((np.abs(np.mgrid[-7:8, -7:8][0]) + np.abs(np.mgrid[-7:8, -7:8][1])) <= 7
            ).astype(np.uint8)


def gray(img):
    a = np.asarray(img, np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def sector_route(warp_color, warp_depth, hole, ref_color, ref_depth, cams, src, dst,
                 n_window=69, sizes=(9, 7, 5, 3), beta=150.0, depth_tol=8.0,
                 bg_bg_only=True, verbose=False, back_depth=None):
    """Fill `hole` in the virtual view by searching the REFERENCE image (sibling's route).

    ``back_depth`` : (H,W) the virtual-view background depth to use for the backward warp of a
    hole pixel.  A hole pixel has no depth of its own, and the sibling's pipeline computes p' from
    the disocclusion's own geometry; falling back to the frame median (the first version) put the
    search window in the wrong place entirely.
    """
    H, W = hole.shape
    out = np.array(warp_color, copy=True)
    known = ~hole
    refc = np.asarray(ref_color, np.float32)
    refg = gray(ref_color)
    warpg = gray(warp_color)
    # reference background mask: the dense background level, eroded by the diamond
    nz = ref_depth[ref_depth > 0]
    sep = float(np.percentile(nz, 60))
    m_bg = ((ref_depth <= sep) & (ref_depth > 0)).astype(np.uint8)
    m_bg = cv2.erode(m_bg, DIAMOND7)
    # write in a priority-ordered sweep (Criminisi: confidence + data term)
    conf = known.astype(np.float32)
    grads = np.abs(cv2.Sobel(warpg, cv2.CV_32F, 1, 0, ksize=3)) + \
        np.abs(cv2.Sobel(warpg, cv2.CV_32F, 0, 1, ksize=3))
    remaining = hole.copy()
    n_filled = 0
    R = n_window // 2
    rmax = max(sizes) // 2
    for _ in range(1_000_000):
        ys, xs = np.nonzero(remaining)
        if ys.size == 0:
            break
        # Criminisi priority: C * D over the frontier closed under the 9x9 patch
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        frontier = remaining & (cv2.dilate(known.astype(np.uint8), k) > 0)
        if not frontier.any():
            frontier = remaining
        fy, fx = np.nonzero(frontier)
        # confidence term
        cf = cv2.boxFilter(conf, -1, (9, 9), normalize=True)[fy, fx]
        df = grads[fy, fx]
        prio = cf * (df + 1e-6)
        i = int(np.argmax(prio))
        py, px = int(fy[i]), int(fx[i])
        # the backward warp of (py, px): its own depth where valid, else the predicted background
        # depth for the hole (a hole pixel has no depth of its own)
        wd = float(warp_depth[py, px])
        if wd <= 0 and back_depth is not None:
            wd = float(back_depth[py, px])
        if wd <= 0:
            wd = float(np.median(ref_depth[ref_depth > 0]))
        u, v, ok = calib.project_pts(np.array([px], np.float64), np.array([py], np.float64),
                                     np.array([max(wd, 1)], np.float64), cams[dst], cams[src])
        wy, wx = int(np.rint(np.nan_to_num(v)[0])), int(np.rint(np.nan_to_num(u)[0]))
        if not ok[0]:
            wy, wx = py, px
        # try sizes largest first
        chosen = None
        for kk in sizes:
            rr = kk // 2
            y0, x0 = py - rr, px - rr
            if y0 < 0 or x0 < 0 or py + rr >= H or px + rr >= W:
                continue
            templ = warpg[y0:y0 + kk, x0:x0 + kk]
            tmask = known[y0:y0 + kk, x0:x0 + kk]
            if tmask.sum() < max(3, kk * kk // 6):
                continue
            sy0 = int(np.clip(wy - R, 0, max(0, H - n_window)))
            sx0 = int(np.clip(wx - R, 0, max(0, W - n_window)))
            cand = m_bg[sy0:sy0 + n_window, sx0:sx0 + n_window]
            if cand.sum() == 0:
                continue
            ch, cw = cand.shape
            # masked SSD on GRAYSCALE (the sibling matches on luma), as an exact correlation
            t = (templ * tmask).astype(np.float32)
            ones = tmask.astype(np.float32)
            reg = refg[sy0:sy0 + n_window, sx0:sx0 + n_window]
            corr = cv2.filter2D(reg, -1, t[::-1, ::-1].copy(), anchor=(kk - 1, kk - 1),
                                borderType=cv2.BORDER_CONSTANT)
            ssum = cv2.filter2D(reg, -1, ones[::-1, ::-1].copy(), anchor=(kk - 1, kk - 1),
                                borderType=cv2.BORDER_CONSTANT)
            tss = float((t * t).sum())
            ssd = tss - 2.0 * corr + ssum
            ssd = np.where(np.isfinite(ssd), ssd, np.inf)
            valid_c = np.zeros((ch, cw), bool)
            valid_c[rr:ch - rr, rr:cw - rr] = cand[rr:ch - rr, rr:cw - rr] > 0
            ssd = np.where(valid_c, ssd, np.inf)
            if depth_tol is not None and wd > 0:
                dl = ref_depth[sy0:sy0 + n_window, sx0:sx0 + n_window].astype(np.float32)
                near = np.abs(dl - wd) <= depth_tol
                ok_c = np.zeros((ch, cw), bool)
                ok_c[rr:ch - rr, rr:cw - rr] = near[rr:ch - rr, rr:cw - rr]
                ssd = np.where(ok_c, ssd, np.inf)
            if not np.isfinite(ssd).any():
                continue
            j = int(np.argmin(ssd))
            cy, cx = np.unravel_index(j, ssd.shape)
            cost = float(ssd[cy, cx]) / max(1, int(tmask.sum()))
            if chosen is None:
                chosen = (kk, sy0 + cy, sx0 + cx, cost)
            if cost <= beta:
                chosen = (kk, sy0 + cy, sx0 + cx, cost)
                break
        if chosen is None:
            # fall back: nearest known pixel
            remain = remaining & ~known
            if remain.any():
                d = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 3)
                ry, rx = np.nonzero(remain)
                jj = int(np.argmin(d[ry, rx]))
                ys2, xs2 = ry[jj], rx[jj]
                out[ys2, xs2] = out[max(0, ys2 - 1), xs2]
                known[ys2, xs2] = True
                remaining[ys2, xs2] = False
                n_filled += 1
            continue
        kk, qy, qx, cost = chosen
        rr = kk // 2
        patch = refc[qy - rr:qy + rr + 1, qx - rr:qx + rr + 1].astype(np.uint8)
        fill = remaining[py - rr:py + rr + 1, px - rr:px + rr + 1]
        out[py - rr:py + rr + 1, px - rr:px + rr + 1][fill] = \
            patch[fill]
        conf[py - rr:py + rr + 1, px - rr:px + rr + 1][fill] = \
            conf[py - rr:py + rr + 1, px - rr:px + rr + 1][fill].mean() if fill.any() else 0
        known[py - rr:py + rr + 1, px - rr:px + rr + 1] |= fill
        remaining[py - rr:py + rr + 1, px - rr:px + rr + 1] &= ~fill
        n_filled += int(fill.sum())
    stats = dict(filled_px=n_filled, leftover=int(remaining.sum()))
    if verbose:
        print(f"  route fill: {n_filled} px, leftover {stats['leftover']}")
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--n_window", type=int, default=69)
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib()
    print(f"{'frame':6s} {'ours':>7s} {'route':>7s} {'delta':>7s} | {'smooth':>16s} "
          f"{'textured':>16s} {'edge':>16s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        if not os.path.isdir(w):
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        wd = np.load(os.path.join(w, "warped_depth.npy")).astype(np.float32)
        wd = np.where(wd < 0, 0, wd)
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        hole = (dis | cr) & ~oofa
        ours = io_utils.imread(os.path.join(d, "60_final", "final.png"))
        # the virtual-view background depth for the hole: the composite depth stage 6 wrote
        comp = np.load(os.path.join(d, "60_final", "final_depth.npy")).astype(np.float32)
        back = np.where(hole, comp, 0.0)
        route, st = sector_route(warp, wd, hole, ref["color"], ref["depth"], cams,
                                 a.src_cam, a.dst_cam, n_window=a.n_window,
                                 back_depth=back, verbose=True)
        p0 = metrics.psnr(ours, gt, mask=hole)
        p1 = metrics.psnr(route, gt, mask=hole)
        g = gray(gt)
        m1 = cv2.boxFilter(g, -1, (7, 7), normalize=True)
        m2 = cv2.boxFilter(g * g, -1, (7, 7), normalize=True)
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge = hole & (cv2.dilate(hole.astype(np.uint8), k) > 0) & \
            (cv2.erode(hole.astype(np.uint8), k) == 0)
        tex = hole & (sd > a.tex_thr)
        sm = hole & ~tex & ~edge

        def pair(mask):
            if not mask.any():
                return "   n/a"
            return f"{metrics.psnr(ours, gt, mask=mask):5.2f}/{metrics.psnr(route, gt, mask=mask):5.2f}"
        print(f"{frame:6s} {p0:7.2f} {p1:7.2f} {p1 - p0:+7.2f} | {pair(sm):>16s} "
              f"{pair(tex):>16s} {pair(edge):>16s}")
        rows.append((p0, p1))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\n  ours  {m(0):6.2f} dB")
        print(f"  route {m(1):6.2f} dB   ({m(1) - m(0):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
