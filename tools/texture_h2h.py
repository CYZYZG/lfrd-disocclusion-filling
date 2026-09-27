"""Texture comparison head-to-head: ours vs the sibling reproduction, same warp, same mask.

    python tools/texture_h2h.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2 --tag ba54

Reads the sibling results written by tools/head_to_head.py
(output/_h2h/<tag>_cam<src><dst>/<frame>_B_noOOFA/03_filled.png) and reports, on the same
disocclusion + crack mask, the LUMA PSNR and the high-frequency energy where the ground truth
is textured -- which is the "背景有纹理" case.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def hf(img):
    return np.abs(cv2.Laplacian(luma(img).astype(np.float32), cv2.CV_32F, ksize=3))


def lpsnr(gt, img, mask):
    d = (luma(gt) - luma(img)) ** 2
    m = np.asarray(mask, bool)
    if m.ndim == 3:
        m = m[..., 0]
    d = d[m]
    return float(10 * np.log10(255.0 ** 2 / d.mean())) if d.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tex_thr", type=float, default=12.0)
    ap.add_argument("--temporal_run", default=None,
                    help="also compare a run that uses the temporal background")
    a = ap.parse_args()
    h2h = os.path.join(io_utils.run_dir("_h2h", create=False),
                       f"{a.tag}_cam{a.src_cam}{a.dst_cam}")
    if not os.path.isdir(h2h):
        h2h = os.path.join(io_utils.run_dir("_h2h", create=False), a.tag)
    if not os.path.isdir(h2h):
        for cand in sorted(os.listdir(io_utils.run_dir("_h2h", create=False))):
            p = os.path.join(io_utils.run_dir("_h2h", create=False), cand)
            if a.tag in cand and os.path.isdir(os.path.join(p, "f000_B_noOOFA")):
                h2h = p
                break
    print(f"{'frame':6s} {'region':>7s} {'tex px':>7s} | {'PSNR ours':>9s} "
          f"{'PSNR sib':>9s} | {'HF ours':>8s} {'HF sib':>7s} {'HF GT':>7s} | "
          f"{'keep ours':>9s} {'keep sib':>8s}")
    agg = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        fp = os.path.join(h2h, f"{frame}_B_noOOFA", "03_filled.png")
        if not os.path.isfile(fp):
            print(f"[skip] {frame}: no {fp}")
            continue
        fd = os.path.join(io_utils.run_dir(a.run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(fd, "20_warp")
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        region = (dis | cr) & ~oofa
        sib = io_utils.imread(fp)
        ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
        st = cv2.boxFilter(luma(gt).astype(np.float32), -1, (7, 7), normalize=True)
        st2 = cv2.boxFilter((luma(gt).astype(np.float32)) ** 2, -1, (7, 7), normalize=True)
        sd = np.sqrt(np.maximum(st2 - st * st, 0))
        tex = region & (sd > a.tex_thr)
        g = hf(gt)[tex].mean() if tex.any() else float("nan")
        ho = hf(ours)[tex].mean() if tex.any() else float("nan")

        def hfs(img):
            # the sibling writes its own hole handling; compare like for like on the same mask
            return hf(img)[tex].mean() if tex.any() else float("nan")
        hs = hfs(sib)
        print(f"{frame:6s} {int(region.sum()):7d} {int(tex.sum()):7d} | "
              f"{lpsnr(gt, ours, region):9.2f} {lpsnr(gt, sib, region):9.2f} | "
              f"{ho:8.1f} {hs:7.1f} {g:7.1f} | {ho / g:9.2f} {hs / g:8.2f}")
        agg.append((int(region.sum()), int(tex.sum()), lpsnr(gt, ours, region),
                    lpsnr(gt, sib, region), ho, hs, g))
    if agg:
        m = lambda i: float(np.nanmean([r[i] for r in agg]))
        print(f"\nMEAN  区域 {m(0):.0f} px（其中纹理 {m(1):.0f} px）")
        print(f"      luma PSNR   ours {m(2):.2f}   sibling {m(3):.2f}   "
              f"(ours - sibling {m(2) - m(3):+.2f} dB)")
        print(f"      纹理区 HF   ours {m(4):.1f}   sibling {m(5):.1f}   GT {m(6):.1f}")
        print(f"      纹理保留率  ours {m(4) / m(6):.2f}   sibling {m(5) / m(6):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
