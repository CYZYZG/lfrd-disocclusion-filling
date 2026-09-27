"""Guard against the photometric stage destroying real structure while chasing PSNR.

The two-stage seam match raises PSNR a lot, but PSNR rewards removing any level/texture error,
including real structure: while iterating, the zoomed result lost the horizontal barre and looked
washed out.  This measures, on the filled region, both the aggregate scores AND the high-frequency
energy against the ground truth, for each strength, so the setting can be chosen with the texture
retention in view rather than only the PSNR.

    python tools/photo_strength_check.py --frames 0,1,2,3,4 --strengths 0,0.25,0.5,0.75,1.0
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics
from lfrd.config import RunConfig

PY = sys.executable
ROOT = r"D:\项目\空洞填补2"


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def hf(img):
    return np.abs(cv2.Laplacian(luma(img).astype(np.float32), cv2.CV_32F, ksize=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,1,2,3,4")
    ap.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--tex_thr", type=float, default=12.0)
    a = ap.parse_args()
    res = {}
    for s in [float(x) for x in a.strengths.split(",")]:
        cfg = RunConfig()
        cfg.temporal_frames = 100
        cfg.photo_strength = s
        cfgp = os.path.join(ROOT, "output", "_cfg2", f"st_{int(s * 100)}.json")
        cfg.save(cfgp)
        run = f"st{int(s * 100)}"
        subprocess.run([PY, os.path.join(ROOT, "run_all.py"), "--run", run, "--pairs", "5:4",
                        "--frames", f"0-{a.frames.split(',')[-1]}", "--config", cfgp],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT)
        rows = []
        for fi in [int(x) for x in a.frames.split(",")]:
            frame = io_utils.frame_name(fi)
            d = os.path.join(io_utils.run_dir(run, create=False), f"cam5-cam4-{frame}")
            if not os.path.isdir(d):
                continue
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 4, frame)["color"]
            img = io_utils.imread(os.path.join(d, "60_final", "final.png"))
            w = os.path.join(d, "20_warp")
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            reg = (dis | cr) & ~oofa
            g = luma(gt)
            m1 = cv2.boxFilter(g.astype(np.float32), -1, (7, 7), normalize=True)
            m2 = cv2.boxFilter((g.astype(np.float32)) ** 2, -1, (7, 7), normalize=True)
            sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
            tex = reg & (sd > a.tex_thr)
            rows.append((metrics.psnr(img, gt, mask=reg),
                         metrics.ssim(img, gt, reg, True),
                         hf(img)[tex].mean() / max(hf(gt)[tex].mean(), 1e-6)))
        res[s] = (float(np.mean([r[0] for r in rows])),
                  float(np.nanmean([r[1] for r in rows])),
                  float(np.mean([r[2] for r in rows])))
    print(f"{'strength':>9s} {'PSNR':>8s} {'SSIM':>8s} {'纹理保留率':>10s}")
    base = res.get(0.0)
    for s, (p, ss, t) in sorted(res.items()):
        print(f"{s:9.2f} {p:8.3f} {ss:8.4f} {t:10.2f}" +
              ("" if base is None or s == 0 else f"   (PSNR {p - base[0]:+.3f}, "
                                                  f"SSIM {ss - base[1]:+.4f})"))
    print("\n纹理保留率 = 空洞纹理区 Laplacian 能量 / 真值同区域；"
          "若它随 strength 明显下降，说明校正把真实结构也压掉了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
