"""Compare stage 6's own accounting between two runs (what actually happened in the pipeline).

A hand-built composite keeps disagreeing with stage 6's output, so read stage 6's own numbers
instead of guessing: how many pixels came from the warped occlusion layer, how many from
postprocessing, and what each region scored.

    python tools/stage6_accounting.py --runs ba54_seq ba54_temporal ba54_latest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补2")
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="f000,f005")
    a = ap.parse_args()
    for run in a.runs:
        print(f"\n===== {run}")
        for frame in a.frames.split(","):
            p = os.path.join(io_utils.run_dir(run, create=False),
                             f"cam{a.src_cam}-cam{a.dst_cam}-{frame}", "60_final",
                             "final_meta.json")
            if not os.path.isfile(p):
                print(f"  {frame}: (missing)")
                continue
            with open(p, encoding="utf-8") as fh:
                m = json.load(fh)
            mr = m.get("metrics_regions", {})
            if not isinstance(mr, dict):
                mr = {}
            def g(k):
                v = mr.get(k)
                return v.get("ours_psnr", float("nan")) if isinstance(v, dict) else float("nan")
            print(f"  {frame}: 从遮挡层 {m.get('filled_from_occlusion_px')} px   "
                  f"后处理 {m.get('postprocess_filled_px')} px   "
                  f"遮挡层无样本 {m.get('disocc_without_sample_px')} px")
            print(f"          PSNR  disocc {g('disocc'):6.2f}  filled {g('filled'):6.2f}  "
                  f"valid {g('valid'):6.2f}  oofa {g('oofa'):6.2f}")
            print(f"          oofa_filled={m.get('oofa_filled')}  leftover={m.get('leftover_px')}")
        # per-frame mean over all 10 frames if available
        d = os.path.join(io_utils.run_dir(run, create=False))
        fr = sorted(x for x in os.listdir(d) if x.startswith(f"cam{a.src_cam}-cam{a.dst_cam}-"))
        took, post = [], []
        for fd in fr:
            p = os.path.join(d, fd, "60_final", "final_meta.json")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    m = json.load(fh)
                took.append(m.get("filled_from_occlusion_px", 0))
                post.append(m.get("postprocess_filled_px", 0))
        if took:
            print(f"  全 {len(took)} 帧均值: 从遮挡层 {np.mean(took):.0f} px, "
                  f"后处理 {np.mean(post):.0f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
