"""Does REAL background content help the sibling too?  Feed it our temporal occlusion layer.

tools/head_to_head.py gives the sibling the plain reference image, while our pipeline gives
itself the *predicted* occlusion layer.  That is not comparable.  This runs the sibling filler
with, in turn,

  A  the plain reference (what head_to_head does)
  B  the reference with the removed band replaced by our single-view occlusion-layer content
  C  the reference with the removed band replaced by the TEMPORAL background model

and scores A/B/C on the same mask.  If C > B, real content helps the sibling as well, and its
head-to-head advantage is about the filling algorithm, not about who has better content.

    python tools/sibling_input_ab.py --src_cam 5 --dst_cam 4 --frames 0,1,2 --tag sibin
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils

SIBLING = r"D:\项目\空洞填补"
PY = sys.executable


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lpsnr(gt, img, mask):
    if not np.asarray(mask, bool).any():
        return float("nan")
    d = ((luma(gt) - luma(img)) ** 2)[mask]
    return float(10 * np.log10(255.0 ** 2 / d.mean()))


def run_sibling(workdir, ref, refdepth, warped, dwarped, hole, gt):
    os.makedirs(workdir, exist_ok=True)
    io_utils.imwrite(os.path.join(workdir, "ref.png"), ref)
    io_utils.imwrite(os.path.join(workdir, "refdepth.png"), refdepth)
    io_utils.imwrite(os.path.join(workdir, "warped.png"), warped)
    io_utils.imwrite(os.path.join(workdir, "dwarped.png"),
                     np.clip(np.where(dwarped < 0, 0, dwarped), 0, 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(workdir, "hole.png"), (hole * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(workdir, "gt.png"), gt)
    out = os.path.join(workdir, "out")
    cmd = [PY, "-m", "viewfill",
           "--image", os.path.join(workdir, "ref.png"),
           "--depth", os.path.join(workdir, "refdepth.png"),
           "--depth-mode", "255",
           "--warped", os.path.join(workdir, "warped.png"),
           "--hole", os.path.join(workdir, "hole.png"),
           "--depth-warped", os.path.join(workdir, "dwarped.png"),
           "--repair-warp", "never",
           "--gt", os.path.join(workdir, "gt.png"),
           "--out", out, "--quiet"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=SIBLING)
    return out, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal_run", default="ba54_temporal")
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", default="sibin")
    a = ap.parse_args()
    root = io_utils.DATASET_ROOT_DEFAULT
    work = os.path.join(io_utils.run_dir("_h2h", create=True), a.tag)
    print(f"{'frame':6s} {'A plain ref':>12s} {'B +our layer':>13s} "
          f"{'C +temporal':>12s} {'C-B':>7s} {'our final':>10s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(a.temporal_run, create=False),
                            f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if not os.path.isdir(base):
            print(f"[skip] {frame}")
            continue
        ref = io_utils.load_view(root, a.src_cam, frame)
        gt = io_utils.load_view(root, a.dst_cam, frame)["color"]
        w = os.path.join(base, "20_warp")
        warped = io_utils.imread(os.path.join(w, "warped_color.png"))
        wd = np.load(os.path.join(w, "warped_depth.npy")).astype(np.float32)
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"),
                             gray=True) > 0
        occ_c = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
        occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        region = dis | cr
        variants = {}
        # A: plain reference
        variants["A"] = (ref["color"], ref["depth"])
        # B: the occlusion layer WITHOUT the temporal substitution -- rerun the temporal run's
        #    stage 5 outputs are already substituted, so use the paper-literal run's layer
        lit = os.path.join(io_utils.run_dir("ba54_seq", create=False),
                           f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        if os.path.isdir(lit):
            variants["B"] = (io_utils.imread(os.path.join(lit, "50_fill",
                                                          "filled_occlusion.png")),
                             io_utils.imread(os.path.join(lit, "50_fill",
                                                          "filled_occlusion_depth.png"),
                                             gray=True))
        variants["C"] = (occ_c, occ_d)
        res = {}
        for name, (rc, rd) in variants.items():
            wd_ = os.path.join(work, f"{frame}_{name}")
            out, p = run_sibling(wd_, rc, rd, warped, wd, region, gt)
            fp = os.path.join(out, "03_filled.png")
            if os.path.isfile(fp):
                res[name] = lpsnr(gt, io_utils.imread(fp), region)
            else:
                tail = (p.stderr or p.stdout or "").strip().splitlines()
                print(f"  [{frame} {name}] failed: {tail[-1][:120] if tail else '?'}")
                res[name] = float("nan")
        ours = lpsnr(gt, io_utils.imread(os.path.join(base, "60_final", "final.png")), region)
        print(f"{frame:6s} {res.get('A', float('nan')):12.2f} "
              f"{res.get('B', float('nan')):13.2f} {res.get('C', float('nan')):12.2f} "
              f"{res.get('C', float('nan')) - res.get('B', float('nan')):+7.2f} {ours:10.2f}")
        rows.append((res.get("A"), res.get("B"), res.get("C"), ours))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\nMEAN  参考项目 + 普通参考图 {m(0):.2f}   + 我们的遮挡层 {m(1):.2f}   "
              f"+ 时序背景 {m(2):.2f}")
        print(f"      (我们的时序最终结果 {m(3):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
