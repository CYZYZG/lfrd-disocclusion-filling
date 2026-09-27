"""Ablation + baseline comparison for the reproduction.

Arms
----
1. `warp_raw`     plain 3D warping of the RAW depth (no ghost preprocessing)
2. `warp_ghost`   3D warping of the PREPROCESSED depth (paper III-A)
3. `direct_inpaint`  Criminisi-style exemplar inpainting applied DIRECTLY on the virtual
                     view (this is the comparison method [17] the paper argues against);
                     it never uses the reference image, so it is the fair "baseline"
                     for the whole contribution.
4. `ours`         the full pipeline (step1..step6 output)

Ghost-removal ablation is measured on a COMMON pixel set (the intersection of the holes of
both warp arms) so that the comparison is not confounded by the two arms having slightly
different hole shapes.

Outputs (under output/<run>/ablation/):
    warp_raw.png, warp_ghost.png, direct_inpaint.png
    report.md, metrics.json, panel_ablation.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import cli, io_utils, metrics, viz
from lfrd import calib, inpaint as inpaint_mod, warp as warp_mod
from lfrd.config import RunConfig


def masked_psnr_ssim(pred, gt, mask, do_ssim=True):
    a = np.array(pred, np.uint8)
    b = np.array(gt, np.uint8)
    m = np.asarray(mask, bool)
    if not m.any():
        return float("nan"), float("nan")
    a[~m] = b[~m]
    return metrics.psnr(a, b), (metrics.ssim(a, b) if do_ssim else float("nan"))


def main():
    ap = cli.base_parser("Ablation and baseline comparison")
    ap.add_argument("--no_direct", action="store_true",
                    help="skip the direct virtual-view inpainting baseline")
    a = ap.parse_args()
    cfg = cli.make_config(a)
    frame = cli.frame_of(a)
    cams = calib.load_calib(cfg.dataset_root, cfg.calib_name)
    src, dst = cfg.src_cam, cfg.dst_cam
    ref = io_utils.load_view(cfg.dataset_root, src, frame)
    gt = io_utils.load_view(cfg.dataset_root, dst, frame)["color"]

    ab = io_utils.run_dir(a.run, create=True)
    ab = os.path.join(ab, "ablation")
    os.makedirs(ab, exist_ok=True)

    # IMPORTANT: the stage directories of a multi-frame run only hold the LAST frame that was
    # processed there.  Every input for this frame must therefore come from the per-frame
    # copy written by run_all.py (`cam<src>-cam<dst>-f###/`), with the stage dir as a
    # fallback for single-frame runs.  Reading the stage dir for frames 0..n-1 silently
    # mixed frames (that produced a bogus -3.4 dB "ghost removal" result).
    fdir = os.path.join(io_utils.run_dir(a.run), f"cam{src}-cam{dst}-{frame}")
    if os.path.isdir(fdir):
        warpd = os.path.join(fdir, "20_warp")
        finald = os.path.join(fdir, "60_final")
    else:
        warpd = io_utils.run_dir(a.run, "warp")
        finald = io_utils.run_dir(a.run, "final")

    # The preprocessed depth is RECOMPUTED for this frame rather than read from
    # `output/<run>/10_preproc/depth_pp.png`: in a multi-frame run that stage file belongs to
    # whichever frame was processed last, so reading it for frames 0..n-1 mixed frames (that
    # produced a bogus -3.4 dB "ghost removal" result).  preprocess_depth is deterministic,
    # so recomputing reproduces exactly what step1 wrote for this frame.
    from lfrd import preprocess
    pprep = preprocess.preprocess_depth(ref["depth"], th=cfg.th, rounds=cfg.ghost_rounds,
                                        return_info=True)
    P_pp = pprep["P_out"]
    print(f"[ablation] preprocessing recomputed for {frame}: marked "
          f"{int(pprep['marked_all'].sum())} px, changed {int((P_pp != ref['depth']).sum())} px")

    arms = {}
    holes = {}
    for name, P in (("warp_raw", ref["depth"]), ("warp_ghost", P_pp)):
        w = warp_mod.warp_view(cams, src, dst, ref["color"], P)
        color = w["warped_color"]
        crack = warp_mod.crack_mask(w["hole"], cfg.crack_max_width)
        if cfg.crack_fill:
            color = warp_mod.fill_cracks(color, crack)
        arms[name] = color
        holes[name] = w["hole"]
        io_utils.imwrite(os.path.join(ab, f"{name}.png"), color)
        print(f"[ablation] {name}: holes {w['hole'].sum()} px "
              f"({100.0 * w['hole'].mean():.2f}%)")

    # common set for the ghost comparison
    common = holes["warp_raw"] & holes["warp_ghost"]
    print(f"[ablation] common hole pixels (both arms): {common.sum()}")

    # 3. direct inpainting of the virtual view (baseline [17])
    if not a.no_direct:
        hd = io_utils.imread(os.path.join(warpd, "hole_disocc.png"), gray=True) > 0
        ho = io_utils.imread(os.path.join(warpd, "hole_oofa.png"), gray=True) > 0
        wd = np.load(os.path.join(warpd, "warped_depth.npy")).astype(np.float32)
        fill_mask = common & ~ho                       # disocclusions + cracks, no OOFA
        depth_u8 = np.clip(np.where(wd < 0, 0, wd), 0, 255).astype(np.uint8)
        base_img = np.array(arms["warp_ghost"], copy=True)
        # a small closing so the baseline sees a continuous hole to fill
        import cv2
        fm = cv2.morphologyEx(fill_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8)) > 0
        res = inpaint_mod.inpaint(base_img, fm, depth_u8,
                                  patch_size=cfg.patch_size, search_w=cfg.search_w,
                                  search_h=cfg.search_h, alpha=cfg.alpha,
                                  depth_tol=cfg.depth_tol, use_bg_term=False,
                                  use_depth_term=True, use_depth_limit=False,
                                  local_search=cfg.local_search,
                                  return_meta=True)
        arms["direct_inpaint"] = res["filled"]
        holes["direct_inpaint"] = fm
        io_utils.imwrite(os.path.join(ab, "direct_inpaint.png"), res["filled"])
        print(f"[ablation] direct_inpaint: {res['n_iters']} iters, "
              f"{res['seconds']:.1f} s, filled {int(fm.sum())} px")

    # 4. ours
    final_path = os.path.join(finald, "final.png")
    if os.path.isfile(final_path):
        arms["ours"] = io_utils.imread(final_path)

    # ---- metrics ---------------------------------------------------------- #
    # The region masks come from THIS frame's per-frame warp copy (falling back to the
    # stage dir for a single-frame run), so the ablation and the main eval/report.md score
    # exactly the same pixels.
    hole_all = io_utils.imread(os.path.join(warpd, "hole_all.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(warpd, "hole_oofa.png"), gray=True) > 0
    disocc_mask = io_utils.imread(os.path.join(warpd, "hole_disocc.png"), gray=True) > 0
    filled = hole_all & ~oofa
    rows = []
    for name, img in arms.items():
        r = {"arm": name}
        r["psnr_whole"] = metrics.psnr(img, gt)
        # the ghost effect lives in the pixels the warp *did* produce, so compare the
        # valid region on the COMMON valid set of both warp arms to avoid confounding
        common_valid = (~holes["warp_raw"]) & (~holes["warp_ghost"])
        r["psnr_valid"], r["ssim_valid"] = masked_psnr_ssim(img, gt, common_valid)
        r["psnr_filled"], r["ssim_filled"] = masked_psnr_ssim(img, gt, filled)
        r["psnr_disocc"], r["ssim_disocc"] = masked_psnr_ssim(img, gt, disocc_mask)
        rows.append(r)
    ghost = {"arm": "ghost_removal"}
    # ghost-removal gain: how well does the warp reproduce the true image on the pixels
    # that BOTH arms warp?  Measured on the common valid set (continuous, no black holes).
    for k, tag in (("warp_raw", "raw"), ("warp_ghost", "ghost")):
        p, s = masked_psnr_ssim(arms[k], gt, common_valid)
        ghost[f"psnr_common_{tag}"] = p
        ghost[f"ssim_common_{tag}"] = s
    ghost["n_common_valid"] = int(common_valid.sum())
    ghost["gain_db"] = ghost.get("psnr_common_ghost", float("nan")) - \
                       ghost.get("psnr_common_raw", float("nan"))
    rows.append(ghost)

    # ---- report ----------------------------------------------------------- #
    lines = [f"# Ablation / baseline — run `{a.run}`, cam{src}->cam{dst} {frame}", ""]
    lines.append("All PSNR/SSIM are against the real capture of cam%d. Region PSNR averages the "
                 "MSE over the region's own pixels (honest); region SSIM uses GT-substituted "
                 "windows and is indicative only." % dst)
    lines.append("")
    lines.append("| arm | whole PSNR | valid PSNR | **filled PSNR** | filled SSIM | "
                 "disocc PSNR |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if r["arm"] == "ghost_removal":
            continue
        lines.append(f"| {r['arm']} | {r['psnr_whole']:.3f} | {r['psnr_valid']:.3f} | "
                     f"**{r['psnr_filled']:.3f}** | {r['ssim_filled']:.4f} | "
                     f"{r['psnr_disocc']:.3f} |")
    lines += ["", "## Ghost-removal ablation", "",
              "Measured on the pixels BOTH warp arms produced (common valid set, "
              f"{ghost['n_common_valid']} px), i.e. where a ghost would be visible:",
              "",
              f"- raw depth      : {ghost.get('psnr_common_raw', float('nan')):.3f} dB / "
              f"SSIM {ghost.get('ssim_common_raw', float('nan')):.4f}",
              f"- preprocessed   : {ghost.get('psnr_common_ghost', float('nan')):.3f} dB / "
              f"SSIM {ghost.get('ssim_common_ghost', float('nan')):.4f}",
              f"- **gain**       : {ghost['gain_db']:+.3f} dB",
              ""]
    if "ours" in arms and "direct_inpaint" in arms:
        ours = [r for r in rows if r["arm"] == "ours"][0]
        base = [r for r in rows if r["arm"] == "direct_inpaint"][0]
        lines += ["## Headline comparison (the paper's claim)", "",
                  f"- direct virtual-view inpainting (baseline [17]): "
                  f"{base['psnr_filled']:.3f} dB / SSIM {base['ssim_filled']:.4f}",
                  f"- proposed (reference-image prediction)        : "
                  f"{ours['psnr_filled']:.3f} dB / SSIM {ours['ssim_filled']:.4f}",
                  f"- **advantage**                                : "
                  f"{ours['psnr_filled'] - base['psnr_filled']:+.3f} dB / "
                  f"{ours['ssim_filled'] - base['ssim_filled']:+.4f}",
                  ""]
    ab_frame = os.path.join(ab, f"cam{src}-cam{dst}-{frame}")
    frame_tag = f"cam{src}-cam{dst}-{frame}"
    os.makedirs(ab_frame, exist_ok=True)
    for name in ("warp_raw", "warp_ghost", "direct_inpaint", "ours"):
        if name in arms:
            io_utils.imwrite(os.path.join(ab_frame, f"{name}.png"), arms[name])
    blob = {"rows": rows, "common_hole_px": int(common.sum()),
            "common_valid_px": int(common_valid.sum()),
            "frame": frame_tag, "config": cfg.to_dict()}
    io_utils.write_text(os.path.join(ab_frame, "report.md"), "\n".join(lines) + "\n")
    io_utils.write_json(os.path.join(ab_frame, "metrics.json"), blob)
    # the run-level files describe the most recent frame (kept for quick inspection)
    io_utils.write_text(os.path.join(ab, "report.md"), "\n".join(lines) + "\n")
    io_utils.write_json(os.path.join(ab, "metrics.json"), blob)
    tiles = [(gt, "ground truth")]
    for name in ("warp_raw", "warp_ghost", "direct_inpaint", "ours"):
        if name in arms:
            tiles.append((arms[name], name))
    viz.panel(tiles, path=os.path.join(ab_frame, "panel_ablation.png"),
              title=f"ablation {frame_tag}")
    viz.panel(tiles, path=os.path.join(ab, "panel_ablation.png"),
              title=f"ablation {frame_tag}")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
