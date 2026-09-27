"""Compare the two reproductions under ONE metric formulation.

D:\\项目\\空洞填补 uses `viz.psnr` = PSNR on **grayscale (luma)**, and reports
`gt_psnr_frame_after` = whole frame with the holes counted as black.  Our own runner uses
**RGB** PSNR and reports a region-restricted value that averages the MSE over the region's
own pixels.  Comparing the two numbers directly is meaningless; this script recomputes OUR
results with THEIR formulation (and theirs-style filled-hole number) so the two are
apples-to-apples.

    python tools/metric_parity.py

Outputs output/_parity/report.md
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def gray(img):
    """Same luma as the sibling project's dibr.viz._gray."""
    a = np.asarray(img, np.float64)
    if a.ndim == 3:
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a


def psnr_gray(gt, img, mask=None, maxval=255.0):
    d = (gray(gt) - gray(img)) ** 2
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == 3:
            m = m[..., 0]
        d = d[m]
    if d.size == 0:
        return float("nan")
    mse = float(d.mean())
    return float("inf") if mse <= 0 else 10.0 * np.log10(maxval ** 2 / mse)


def psnr_rgb(gt, img, mask=None, maxval=255.0):
    d = (np.asarray(gt, np.float64) - np.asarray(img, np.float64)) ** 2
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == 3:
            m = m[..., 0]
        d = d[m]
        if d.size == 0:
            return float("nan")
        mse = float(d.mean())
    else:
        mse = float(d.mean())
    return float("inf") if mse <= 0 else 10.0 * np.log10(maxval ** 2 / mse)


# the sibling project's reported averages (output/step8_eval/report.md)
REFERENCE = {
    ("6", "7"): {"pair": "cam6→cam7", "frames": 10, "psnr": 27.53, "ssim": 0.8244,
                 "holes": 15.97, "fill_px_psnr": 23.89},
    ("6", "5"): {"pair": "cam6→cam5", "frames": 10, "psnr": 27.00, "ssim": 0.8082,
                 "holes": 15.32, "fill_px_psnr": 21.11},
    ("3", "0"): {"pair": "cam3→cam0", "frames": 10, "psnr": 22.98, "ssim": 0.7214,
                 "holes": 33.00, "fill_px_psnr": 20.08},
    ("3", "2"): {"pair": "cam3→cam2", "frames": 10, "psnr": 26.42, "ssim": 0.8001,
                 "holes": 17.04, "fill_px_psnr": 20.27},
}

OUR_RUNS = [("q_67", 6, 7), ("q_65", 6, 5), ("q_30", 3, 0), ("q_32", 3, 2)]


def main():
    out = io_utils.run_dir("_parity", create=True)
    lines = ["# Metric parity: same results, two formulations", "",
             "`holes` = our full hole mask (cracks + disocclusion + OOFA), black in the warp.",
             "`valid` = pixels the warp produced. OOFA is never filled by either side, so it",
             "is black in the GT comparison for both.", "",
             "| pair | frames | our holes % | our whole PSNR (luma) | sibling whole PSNR "
             "(luma, its own mask) | **our filled-hole PSNR (luma)** | **sibling filled-hole** | delta |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run, src, dst in OUR_RUNS:
        key = (str(src), str(dst))
        ref = REFERENCE.get(key)
        acc = {k: [] for k in ("hole", "rgb", "wholegray", "fillgray")}
        nframes = 0
        for fd in sorted(glob.glob(os.path.join(io_utils.run_dir(run, create=False),
                                                f"cam{src}-cam{dst}-f*"))):
            frame = fd.split("-")[-1]
            fin = os.path.join(fd, "60_final", "final.png")
            w = os.path.join(fd, "20_warp")
            hp = os.path.join(w, "hole_all.png")
            if not (os.path.isfile(fin) and os.path.isfile(hp)):
                continue
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
            ours = io_utils.imread(fin)
            hole = io_utils.imread(hp, gray=True) > 0
            oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
            dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
            add = hole & ~oofa if os.path.isfile(os.path.join(w, "hole_oofa.png")) else hole
            acc["hole"].append(100.0 * hole.mean())
            acc["rgb"].append(psnr_rgb(gt, ours))
            acc["wholegray"].append(psnr_gray(gt, ours))
            acc["fillgray"].append(psnr_gray(gt, ours, mask=add))
            nframes += 1
        if not nframes:
            continue
        m = lambda k: float(np.mean(acc[k]))
        rp = ref['fill_px_psnr'] if ref else float('nan')
        lines.append(
            f"| {ref['pair'] if ref else f'cam{src}→cam{dst}'} | {nframes} | "
            f"{m('hole'):.2f} | {m('wholegray'):.2f} | "
            f"{ref['psnr'] if ref else float('nan'):.2f} | "
            f"**{m('fillgray'):.2f}** | {rp:.2f} | {m('fillgray') - rp:+.2f} |")
    lines += ["", "## Reading", "",
              "- `our whole PSNR (luma)` vs `sibling whole PSNR (luma)` is the only honest",
              "  whole-frame comparison.  A gap here means one of the two warps/img is better.",
              "- `filled-hole PSNR` compares only the pixels each method actually had to",
              "  synthesise, which isolates the hole filling from everything else."]
    io_utils.write_text(os.path.join(out, "report.md"), "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
