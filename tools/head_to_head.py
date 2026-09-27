"""Head-to-head: run the sibling project's filler on OUR warp, compare like-for-like.

The sibling project (`viewfill`; set SIBLING_ROOT to its checkout) fills an already-warped view.  Feeding it
exactly the warp OUR pipeline produced (same colour, same depth, same hole mask) removes every
confound except the filling algorithm itself, and the two results can then be scored on ONE
mask with ONE metric:

    fillable region = disocclusion + cracks   (what the paper is responsible for)
    OOFA            = excluded (unreachable; the sibling fills it, this paper does not)

Two variants of the input are produced:
  A) hole = crack + disocclusion + OOFA   -- the sibling's own convention (it fills OOFA)
  B) hole = crack + disocclusion          -- the current paper's convention (OOFA stays black)

    python tools/head_to_head.py --run q_67 --src_cam 6 --dst_cam 7 --frames 0,1,2
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils

SIBLING = os.environ.get("SIBLING_ROOT")
if not SIBLING:
    raise SystemExit(
        "set SIBLING_ROOT to the sibling DIBR reproduction, or pass --sibling")
SIBLING = str(SIBLING)
PY = sys.executable


def gray(a):
    a = np.asarray(a, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def psnr(gt, img, mask):
    d = (gray(gt) - gray(img)) ** 2
    m = np.asarray(mask, bool)
    if m.ndim == 3:
        m = m[..., 0]
    d = d[m]
    return float(10 * np.log10(255.0 ** 2 / d.mean())) if d.size else float("nan")


def ssim_gray(gt, img, mask):
    """SSIM on luma with an 11x11 Gaussian window (same as lfrd.metrics.ssim, 1 channel)."""
    from scipy.ndimage import gaussian_filter
    x, y = gray(gt), gray(img)
    sig, c1, c2 = 1.5, (0.01 * 255) ** 2, (0.03 * 255) ** 2
    w = lambda z: gaussian_filter(z, sig, mode="reflect", truncate=3.0)
    mux, muy = w(x), w(y)
    s = ((2 * mux * muy + c1) * (2 * (w(x * y) - mux * muy) + c2)) / \
        ((mux ** 2 + muy ** 2 + c1) * (w(x * x) - mux ** 2 + w(y * y) - muy ** 2 + c2) + 1e-12)
    m = np.asarray(mask, bool)
    if m.ndim == 3:
        m = m[..., 0]
    return float(s[m].mean()) if m.any() else float("nan")


def run_sibling(workdir, warped, hole, dwarped, ref, refdepth, out, gt):
    os.makedirs(workdir, exist_ok=True)
    io_utils.imwrite(os.path.join(workdir, "warped.png"), warped)
    io_utils.imwrite(os.path.join(workdir, "hole.png"), (hole * 255).astype(np.uint8))
    np.save(os.path.join(workdir, "dwarped.npy"), dwarped.astype(np.int16))
    io_utils.imwrite(os.path.join(workdir, "ref.png"), ref)
    io_utils.imwrite(os.path.join(workdir, "refdepth.png"), refdepth)
    io_utils.imwrite(os.path.join(workdir, "gt.png"), gt)
    # the sibling CLI reads PNG depth through its own loader; write the warped depth as PNG
    io_utils.imwrite(os.path.join(workdir, "dwarped.png"),
                     np.clip(np.where(dwarped < 0, 0, dwarped), 0, 255).astype(np.uint8))
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
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"{a.run}_cam{a.src_cam}{a.dst_cam}"
    frames = [int(x) for x in a.frames.split(",")]
    outroot = io_utils.run_dir("_h2h", create=True)
    rows = []
    for fi in frames:
        frame = io_utils.frame_name(fi)
        fd = os.path.join(io_utils.run_dir(a.run, create=False),
                          f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(fd, "20_warp")
        if not os.path.isdir(w):
            print(f"[skip] {frame}: no artefacts")
            continue
        ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.src_cam, frame)
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        warped = io_utils.imread(os.path.join(w, "warped_color.png"))
        wd = np.load(os.path.join(w, "warped_depth.npy")).astype(np.float32)
        hole_all = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        crack = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
        fill_region = dis | crack
        valid = ~hole_all

        rec = {"frame": frame, "hole_all_px": int(hole_all.sum()),
               "fill_region_px": int(fill_region.sum()),
               "ours_filled": psnr(gt, ours, fill_region),
               "ours_filled_ssim": ssim_gray(gt, ours, fill_region),
               "ours_valid": psnr(gt, ours, valid),
               "warp_filled": psnr(gt, warped, fill_region)}
        for variant, hole_in in (("A_withOOFA", hole_all), ("B_noOOFA", fill_region)):
            work = os.path.join(outroot, tag, f"{frame}_{variant}_in")
            out = os.path.join(outroot, tag, f"{frame}_{variant}")
            p = run_sibling(work, warped, hole_in, wd, ref["color"], ref["depth"], out, gt)
            res = os.path.join(out, "03_filled.png")
            if not os.path.isfile(res):
                rec[f"h2h_{variant}"] = float("nan")
                print(f"[warn] {frame} {variant}: no filled.png (exit {p.returncode})")
                if p.stderr:
                    print("   ", p.stderr.strip().splitlines()[-1][:200])
                continue
            sib = io_utils.imread(res)
            rec[f"h2h_{variant}"] = psnr(gt, sib, fill_region)
            rec[f"h2h_{variant}_ssim"] = ssim_gray(gt, sib, fill_region)
            rec[f"h2h_{variant}_valid"] = psnr(gt, sib, valid)
        rows.append(rec)
        print(f"{frame}: holes {rec['hole_all_px']}  ours(filled) {rec['ours_filled']:.2f}  "
              f"sibling(A) {rec.get('h2h_A_withOOFA', float('nan')):.2f}  "
              f"sibling(B) {rec.get('h2h_B_noOOFA', float('nan')):.2f}")
    # summary
    def m(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")
    lines = [f"# Head-to-head on the SAME warp — {tag}", "",
             "Both methods fill the identical warp (same colour, depth and hole mask), so the",
             "only difference is the filling algorithm.  Metric: luma PSNR over the fillable",
             "region (disocclusion + cracks).", "",
             "| frame | warp | ours | sibling (hole=all incl. OOFA) | sibling (hole=disocc+crack) |",
             "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['frame']} | {r['warp_filled']:.2f} | **{r['ours_filled']:.2f}** | "
                     f"{r.get('h2h_A_withOOFA', float('nan')):.2f} | "
                     f"{r.get('h2h_B_noOOFA', float('nan')):.2f} |")
    lines += ["", "| mean | warp 7 dB-ish | **%s** | %s | %s |" %
              (f"{m('ours_filled'):.2f}", f"{m('h2h_A_withOOFA'):.2f}",
               f"{m('h2h_B_noOOFA'):.2f}"), ""]
    txt = "\n".join(lines) + "\n"
    io_utils.write_text(os.path.join(outroot, tag, "report.md"), txt)
    io_utils.write_json(os.path.join(outroot, tag, "metrics.json"), rows)
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
