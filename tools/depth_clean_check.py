"""Clean depth/geometry diagnostic for the temporal background substitution.

The previous attempt reported median 0 for the model depth, which means it was averaging over
pixels where the model has no evidence (depth 0).  This restricts every statistic to the pixels
where the model actually contributes, and reports the comparison that matters:

    z_model  the temporal background depth, on the pixels it covers
    z_pred   stage 4's per-row predicted background depth, on THE SAME pixels
    z_ref    the reference depth of that pixel in the target frame

For a STATIC camera, z_ref at a pixel that is background in the target frame is the truth for
that ray (and the sample the pipeline uses), so |z_model - z_ref| measures the model's own
consistency, and |z_pred - z_ref| is the same for the single-view prediction.

    python tools/depth_clean_check.py --src_cam 5 --dst_cam 4 --frames f000,f005
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, temporal


def stats(name, v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return f"{name}: (none)"
    return (f"{name}: n={v.size} median {np.median(v):.1f} mean {v.mean():.1f} "
            f"p10 {np.percentile(v, 10):.1f} p90 {np.percentile(v, 90):.1f} "
            f"zeros {100 * (v == 0).mean():.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="f000,f005")
    ap.add_argument("--n_frames", type=int, default=100)
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    for frame in a.frames.split(","):
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(d):
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
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
            root, a.src_cam, list(range(a.n_frames)), rm, q=10.0, tol=4.0, ref_level=lvl)
        valid = model["valid"]
        zt = model["bg_depth"].astype(np.float64)
        ns = model["n_samples"]
        print(f"\n===== {frame}   移除区 {int(rm.sum())} px，模型覆盖 "
              f"{int(valid.sum())} px ({100 * valid.mean():.1f}% of frame)")
        print("  " + stats("模型深度(仅覆盖处)", zt[valid]))
        print("  " + stats("模型深度(覆盖处对应的 stage4 预测)",
                            pred[valid]))
        print("  " + stats("参考图深度(覆盖处)", ref["depth"].astype(np.float64)[valid]))
        bgpix = valid & ~fg & (ref["depth"] > 0)
        print(f"  其中「目标帧里本身就是背景」的像素 {int(bgpix.sum())} px：")
        print("  " + stats("  模型深度", zt[bgpix]))
        print("  " + stats("  stage4 预测", pred[bgpix]))
        print("  " + stats("  参考图真实背景深度", ref["depth"].astype(np.float64)[bgpix]))
        if bgpix.any():
            t = np.abs(zt[bgpix] - ref["depth"].astype(np.float64)[bgpix])
            p = np.abs(pred[bgpix] - ref["depth"].astype(np.float64)[bgpix])
            print(f"  |模型 - 真实背景| 中位 {np.median(t):.1f}    "
                  f"|stage4预测 - 真实背景| 中位 {np.median(p):.1f}")
        print("  " + stats("模型时间样本数(覆盖处)", ns[valid].astype(np.float64)))
        print("  " + stats("模型时间样本数(全移除区)", ns[rm].astype(np.float64)))
        if lvl is not None:
            print("  " + stats("stage4 背景水平图(移除区)", np.asarray(lvl, np.float64)[rm]))
        print("  " + stats("参考图前景掩码覆盖率(模型覆盖处)",
                            fg[valid].astype(np.float64) * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
