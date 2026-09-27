"""How much would a better depth field for the predicted occlusion layer buy?

The occlusion-layer DEPTH is what positions the synthesised texture inside the hole.  This
script takes the depth prediction of one frame, produces several candidate refinements, and
scores each one the way it is actually used: warp the (unchanged) occlusion colour with that
depth into the virtual view, copy it onto the disocclusions, and measure the resulting PSNR.

Candidates
    pred        the current per-row prediction (paper eq. 4-5)
    smoothK     convex smoothing of the hole depth along rows (K passes) -- tests whether the
                piecewise-constant field is too coarse
    localfit    per-run least-squares plane fitted to the surrounding BACKGROUND pixels,
                clamped between the two background levels -- estimates a slanted surface
    gt          oracle: the real reference depth inside the removed region

    python tools/depth_refine_probe.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, warp as W


def runs_of(row):
    xs = np.nonzero(row)[0]
    if xs.size == 0:
        return []
    out, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x != p + 1:
            out.append((int(s), int(p)))
            s = x
        p = x
    out.append((int(s), int(p)))
    return out


def smooth_hole(depth, mask, passes=1, k=5):
    """Convex smoothing along rows inside the hole, keeping the non-hole depth untouched."""
    out = depth.astype(np.float32).copy()
    for _ in range(passes):
        pad = np.pad(out, ((0, 0), (k // 2, k // 2)), mode="edge")
        acc = np.zeros_like(out)
        for i in range(k):
            acc += pad[:, i:i + out.shape[1]]
        sm = np.where(mask, acc / float(k), out)
        out = sm
    return out


def local_plane_fit(depth, mask, bg_level_map=None, halfwin=3):
    """Fit a plane to the surrounding background pixels of each connected removal run.

    For every run the background samples are the non-hole pixels within a vertical band
    around the run on the same rows plus the two adjacent background columns, and the fitted
    plane is evaluated inside the run.
    """
    out = depth.astype(np.float32).copy()
    H, W = depth.shape
    valid = (depth > 0) & ~mask
    for v in range(H):
        for (x0, x1) in runs_of(mask[v]):
            # background samples: rows v-halfwin..v+halfwin, columns just outside the run
            r0, r1 = max(0, v - halfwin), min(H, v + halfwin + 1)
            c0, c1 = max(0, x0 - 8), min(W, x1 + 9)
            ys, xs = np.mgrid[r0:r1, c0:c1]
            sel = valid[r0:r1, c0:c1] & ((xs < x0) | (xs > x1))
            if sel.sum() < 8:
                continue
            A = np.stack([xs[sel], ys[sel], np.ones(sel.sum())], 1).astype(np.float64)
            b = depth[r0:r1, c0:c1][sel].astype(np.float64)
            try:
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            u = np.arange(x0, x1 + 1, dtype=np.float64)
            pred = coef[0] * u + coef[1] * v + coef[2]
            lo, hi = float(b.min()), float(b.max())
            out[v, x0:x1 + 1] = np.clip(pred, lo, hi)
    return out


def score(cams, src, dst, gt_color, dis, occ_color, depth_field, cache={}):
    r = W.warp_view(cams, src, dst, occ_color, np.clip(np.rint(depth_field), 0, 255)
                    .astype(np.uint8))
    take = dis & ~r["hole"]
    img = np.array(gt_color, copy=True)      # with GT elsewhere: region-only comparison
    img[take] = r["warped_color"][take]
    if not take.any():
        return float("nan"), 0
    return metrics.psnr(img, gt_color, mask=take), int(take.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    a = ap.parse_args()
    cams = calib.load_calib()
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            continue
        gt_view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)
        ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.src_cam, frame)
        gt = gt_view["color"]
        occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
        occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        pred = io_utils.imread(os.path.join(base, "40_removal", "depth_pred.png"),
                               gray=True).astype(np.float32)
        rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"),
                              gray=True) > 0
        # the occlusion colour has the removed region inpainted; use our predicted depth there
        cand = {}
        cand["pred"] = occ_d.astype(np.float32)
        for k in (1, 3, 6):
            cand[f"smooth{k}"] = smooth_hole(pred, rm, passes=k, k=5)
        # splice the refinement into the full reference depth (outside the removed region the
        # occlusion depth is the original reference depth)
        for name in list(cand):
            if name == "pred":
                continue
            full = np.array(occ_d, dtype=np.float32)
            full[rm] = np.clip(cand[name][rm], 0, 255)
            cand[name] = full
        cand["localfit"] = np.array(occ_d, dtype=np.float32)
        cand["localfit"][rm] = np.clip(local_plane_fit(pred, rm)[rm], 0, 255)
        cand["gt"] = np.array(occ_d, dtype=np.float32)
        cand["gt"][rm] = ref["depth"][rm]      # oracle: true reference depth in the removed band
        rec = {"frame": frame}
        for name, d in cand.items():
            p, n = score(cams, a.src_cam, a.dst_cam, gt, dis, occ_c, d)
            rec[name] = p
            rec[name + "_px"] = n
        rows.append(rec)
        print(f"{frame}: " + "  ".join(f"{k}={rec[k]:.2f}" for k in
                                       ("pred", "smooth1", "smooth3", "smooth6",
                                        "localfit", "gt")))
    if rows:
        print("\nMEAN: " + "  ".join(
            f"{k}={np.mean([r[k] for r in rows]):.2f}"
            for k in ("pred", "smooth1", "smooth3", "smooth6", "localfit", "gt")))
        print(f"covered px/frame: {int(np.mean([r['pred_px'] for r in rows]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
