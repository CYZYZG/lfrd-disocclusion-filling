"""Render the result galleries the user asked for.

Produces, for a set of frames of one BA54 run:

  <run>/panels/gallery_frames.png    per-frame 4-up: GT / plain warp / paper-literal /
                                     best (temporal + OOFA), with the hole region marked
  <run>/panels/gallery_zoom_<f>.png  zoom of the largest disocclusion, 5-up including the
                                     predicted occlusion layer
  <run>/panels/gallery_metrics.png   per-frame PSNR/SSIM bars for the three configurations

    python tools/make_gallery.py --best ba54_latest --baseline ba54_seq --frames 0,3,5,8
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, viz


def load(run, src, dst, frame):
    d = os.path.join(io_utils.run_dir(run, create=False), f"cam{src}-cam{dst}-{frame}")
    w = os.path.join(d, "20_warp")
    if not os.path.isdir(w):
        return None
    return dict(
        dir=d, warp=io_utils.imread(os.path.join(w, "warped_color.png")),
        dis=io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0,
        cr=io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0,
        oofa=io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0,
        final=io_utils.imread(os.path.join(d, "60_final", "final.png")),
        occ=io_utils.imread(os.path.join(d, "50_fill", "filled_occlusion.png")),
    )


def region_psnr(gt, img, mask):
    return metrics.psnr(img, gt, mask=mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="ba54_latest")
    ap.add_argument("--baseline", default="ba54_seq")
    ap.add_argument("--src_cam", type=int, default=5)
    ap.add_argument("--dst_cam", type=int, default=4)
    ap.add_argument("--frames", default="0,3,5,8")
    a = ap.parse_args()
    outdir = os.path.join(io_utils.run_dir(a.best, create=True), "panels")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        b = load(a.best, a.src_cam, a.dst_cam, frame)
        o = load(a.baseline, a.src_cam, a.dst_cam, frame)
        if b is None or o is None:
            print(f"[skip] {frame}")
            continue
        reg = (b["dis"] | b["cr"]) & ~b["oofa"]
        p_plain = region_psnr(gt, b["warp"], reg)
        p_lit = region_psnr(gt, o["final"], reg)
        p_best = region_psnr(gt, b["final"], reg)
        # mask overlay on the best result: green = filled, blue = OOFA untouched
        mark = np.array(b["final"], copy=True)
        filled = np.zeros(reg.shape, bool)
        filled[reg] = True
        mark[filled] = (0.55 * mark[filled] +
                        0.45 * np.array([0, 255, 0], np.uint8)).astype(np.uint8)
        oo = b["oofa"]
        mark[oo] = (0.6 * mark[oo] + 0.4 * np.array([80, 140, 255], np.uint8)).astype(np.uint8)
        rows.append((frame, gt, b, o, p_plain, p_lit, p_best, mark))
        print(f"{frame}: plain {p_plain:.2f}  paper-literal {p_lit:.2f}  best {p_best:.2f}")

    # ---- gallery: one row of 4 per frame --------------------------------------- #
    tiles = []
    for (frame, gt, b, o, p_plain, p_lit, p_best, mark) in rows:
        tiles += [
            (gt, f"{frame} GT (cam{a.dst_cam})"),
            (b["warp"], f"{frame} plain warp  {p_plain:.2f} dB"),
            (o["final"], f"{frame} paper-literal  {p_lit:.2f} dB"),
            (b["final"], f"{frame} LATEST  {p_best:.2f} dB  (+{p_best - p_lit:.2f})"),
        ]
    p = os.path.join(outdir, "gallery_frames.png")
    viz.grid(tiles, cols=4, height=215, path=p,
             title="BA54 cam5->cam4 — 空洞区 PSNR（每行一帧）")
    print("written:", p)

    # ---- zoom of the largest disocclusion for each frame ----------------------- #
    for (frame, gt, b, o, p_plain, p_lit, p_best, mark) in rows:
        n, lab, st, _ = cv2.connectedComponentsWithStats(b["dis"].astype(np.uint8), 8)
        i = 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h = [int(v) for v in st[i][:4]]
        cx, cy = x + w // 2, y + h // 2
        zw, zh = max(150, int(w * 1.9)), max(150, int(h * 0.42))
        z = lambda im: viz.zoom(im, cx, cy, zw, zh, 470, 360)
        tz = [(z(gt), "GT"), (z(b["warp"]), "plain warp"),
              (z(o["occ"]), "occlusion layer (paper-literal)"),
              (z(b["occ"]), "occlusion layer (+temporal, real content)"),
              (z(o["final"]), f"final paper-literal {p_lit:.2f} dB"),
              (z(b["final"]), f"final LATEST {p_best:.2f} dB")]
        pz = os.path.join(outdir, f"gallery_zoom_{frame}.png")
        viz.grid(tz, cols=3, height=300, path=pz,
                 title=f"BA54 {frame} — 最大空洞放大（背景纹理区）")
        print("written:", pz)

    # ---- metrics bars ---------------------------------------------------------- #
    try:
        import matplotlib                                            # noqa: F401
        have_mpl = True
    except Exception:                                                # noqa: BLE001
        have_mpl = False
    lines = ["# BA54 cam5->cam4 最新结果（逐帧，空洞区 luma PSNR）", "",
             "| frame | plain warp | paper-literal | **LATEST (temporal+OOFA)** | 提升 |",
             "| --- | --- | --- | --- | --- |"]
    for (frame, gt, b, o, p_plain, p_lit, p_best, mark) in rows:
        lines.append(f"| {frame} | {p_plain:.2f} | {p_lit:.2f} | **{p_best:.2f}** | "
                     f"+{p_best - p_lit:.2f} |")
    if rows:
        m = lambda i: float(np.mean([r[i] for r in rows]))
        lines += ["", f"**均值：plain {m(4):.2f} → paper-literal {m(5):.2f} → "
                      f"LATEST {m(6):.2f} dB（+{m(6) - m(5):.2f}）**", ""]
    io_utils.write_text(os.path.join(outdir, "gallery_metrics.md"), "\n".join(lines) + "\n")
    print("written:", os.path.join(outdir, "gallery_metrics.md"))
    print("mark overlay skipped" if not have_mpl else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
