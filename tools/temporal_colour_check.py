"""Is the temporal model's COLOUR right, or is it averaging across an inconsistent background?

For a static camera the background colour at a pixel should be the same in every frame where it
is background, so the temporal model's output can be checked against the reference image
directly: for each covered pixel, compare the model colour with (a) the reference colour in the
target frame, and (b) the median reference colour over the frames the model actually sampled.

If (b) agrees but (a) does not, the model is internally consistent but that background simply
changed over the sequence (lighting / the curtain moving) -- then the answer is to restrict the
model to frames close in time, not to change the geometry.

    python tools/temporal_colour_check.py --src_cam 5 --dst_cam 4 --frames f000,f005
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, temporal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="f000,f005")
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--window", type=int, default=10,
                    help="also test restricting the model to +/- this many frames")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    print(f"{'frame':6s} {'cover':>7s} | {'vs ref(target)':>14s} {'vs ref(median)':>15s} "
          f"{'frame-to-frame':>14s} | {'near-window':>11s} {'vs ref(target)':>14s}")
    for frame in a.frames.split(","):
        fi = int(frame[1:])
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(d):
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        rm = io_utils.imread(os.path.join(d, "40_removal", "removed_mask.png"), gray=True) > 0
        meta = None
        try:
            meta = io_utils.load_npz(os.path.join(d, "40_removal", "removal_meta.npz"))
        except Exception:                                              # noqa: BLE001
            pass
        lvl = meta.get("bg_level") if isinstance(meta, dict) else None
        m = temporal.build_temporal_background(
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=lvl)
        valid = m["valid"]
        if not valid.any():
            continue
        ys, xs = np.nonzero(valid)
        col = m["bg_color"][ys, xs].astype(np.float64)
        # (a) reference colour in the target frame at that pixel
        col_ref = ref["color"][ys, xs].astype(np.float64)
        # (b) median reference colour over the frames the model sampled
        P = np.stack([io_utils.imread(os.path.join(root, f"cam{a.src_cam}",
                                                   f"depth-cam{a.src_cam}-f{f:03d}.png"),
                                      gray=True)[ys, xs] for f in range(a.n_frames)])
        z = np.percentile(P, 10, axis=0)
        near = (P <= (z[None, :] + 4.0)) & (P > 0)
        samples = []
        for f in range(a.n_frames):
            sel = near[f]
            if sel.any():
                cc = io_utils.imread(os.path.join(root, f"cam{a.src_cam}",
                                                  f"color-cam{a.src_cam}-f{f:03d}.jpg"))
                samples.append((f, cc[ys, xs].astype(np.float64), sel))
        med = np.zeros_like(col)
        cnt = np.zeros(ys.size, np.float64)
        vals_by_pixel = [[] for _ in range(0)]
        # stream: gather per-pixel sample lists only for the median (memory-heavy otherwise)
        stack = []
        for f, cc, sel in samples:
            stack.append((f, cc, sel))
        per_pixel = np.full((len(stack), ys.size, 3), np.nan)
        for i, (f, cc, sel) in enumerate(stack):
            per_pixel[i][sel] = cc[sel]
        with np.errstate(all="ignore"):
            medc = np.nanmedian(per_pixel, 0)
        ok = ~np.isnan(medc[:, 0])
        # frame-to-frame variability of the sampled colours
        var = np.nanstd(per_pixel, 0)[ok].mean()
        # near-window variant
        lo, hi = max(0, fi - a.window), min(a.n_frames, fi + a.window + 1)
        idx = [i for i, (f, _, _) in enumerate(stack) if lo <= f < hi]
        if idx:
            with np.errstate(all="ignore"):
                medn = np.nanmedian(per_pixel[idx], 0)
            d_near = np.abs(medn[ok] - col_ref[ok]).mean()
        else:
            d_near = float("nan")
        d_ref = np.abs(col - col_ref).mean()
        d_med = np.abs(col[ok] - medc[ok]).mean() if ok.any() else float("nan")
        print(f"{frame:6s} {int(valid.sum()):7d} | {d_ref:14.2f} {d_med:15.2f} "
              f"{var:14.2f} | {d_near:11.2f} {d_near:14.2f}")
    print("\nvs ref(target) = 模型颜色 与 目标帧参考图在该像素的颜色 的平均绝对差")
    print("vs ref(median) = 与 模型实际采样的那些帧的中位颜色 的差（衡量模型自身是否自洽）")
    print("frame-to-frame = 被采样帧之间颜色的标准差（背景是否随时间变化）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
