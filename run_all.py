"""End-to-end driver + evaluation.

Runs the reproduction pipeline for one or more frames / camera pairs, compares the result
against the real capture of the virtual camera, and writes an evaluation report.

    python run_all.py --run ba54 --src_cam 5 --dst_cam 4 --frames 0-9
    python run_all.py --run ba54 --frames 0-9 --skip_existing
    python run_all.py --run ba54 --frames 0-9 --ablation

Arms reported per frame (whole frame, disocclusion region, filled pixels):
    warp_only      plain 3D warping (no preprocessing, no filling)   -- lower bound
    warp_ghost     3D warping of the preprocessed depth              -- + ghost removal
    inpaint_direct virtual-view inpainting of the hole (Criminisi baseline, optional)
    ours           full pipeline                                      -- final
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import cli, io_utils, metrics, viz
from lfrd.config import RunConfig, parse_frames

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = ["step1_preprocess.py", "step2_warp.py", "step3_classify.py",
         "step4_removal.py", "step5_inpaint.py", "step6_render.py"]


def run_step(script, args, log_path):
    cmd = [PY, os.path.join(ROOT, script)] + args
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    dt = time.time() - t0
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}\n[exit {p.returncode} in {dt:.1f}s]\n")
    return p.returncode, p.stdout, dt


def main():
    ap = cli.base_parser("Run the full reproduction pipeline and evaluate it")
    ap.add_argument("--pairs", default=None, help="e.g. 5:4,5:6,6:7 (overrides --src_cam/--dst_cam)")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="also compute the warp-only / no-ghost arms")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--eval_only", action="store_true",
                    help="no pipeline run: re-evaluate every frame directory that already "
                         "exists under output/<run>/ (needed after a multi-frame run, which "
                         "writes all frames into the same stage dirs)")
    ap.add_argument("--continue_on_fail", action="store_true",
                    help="keep running the remaining stages after a stage's internal checks "
                         "fail (the failure count is still reported and the frame is flagged; "
                         "used to MEASURE pairs whose internal invariants are not met yet)")
    ap.add_argument("--fill_oofa", action="store_true",
                    help="pass --fill_oofa to step6: also fill the out-of-field area. The "
                         "paper's figures leave it black, but every competing implementation "
                         "fills it, so turn this on when comparing whole-frame PSNR.")
    a = ap.parse_args()

    cfg = cli.make_config(a)
    pairs = []
    if a.pairs:
        for p in a.pairs.split(","):
            s, d = p.split(":")
            pairs.append((int(s), int(d)))
    else:
        pairs = [(cfg.src_cam, cfg.dst_cam)]

    frames = parse_frames(a.frames) if a.frames else cfg.frames
    run_root = io_utils.run_dir(a.run)
    log_path = os.path.join(run_root, "run.log")
    os.makedirs(io_utils.run_dir(a.run, "eval"), exist_ok=True)
    cfg.save_to_run(a.run)

    rows = []
    if a.eval_only:
        # per-frame dirs are named cam<src>-cam<dst>-f<idx> (the stage dirs themselves hold
        # only the last frame processed, so a multi-frame run must be scored per frame dir)
        pairs = sorted({(int(d.split("-")[0][3:]), int(d.split("-")[1][3:]))
                        for d in os.listdir(run_root)
                        if d.startswith("cam") and os.path.isdir(os.path.join(run_root, d))})
        for (src, dst) in pairs:
            for fi in range(100):
                frame = io_utils.frame_name(fi)
                fdir_frame = os.path.join(run_root, f"cam{src}-cam{dst}-{frame}")
                if not os.path.isdir(fdir_frame):
                    continue
                row = evaluate(a.run, src, dst, fi, cfg, ablation=a.ablation,
                               frame_dir=fdir_frame)
                if row:
                    row["pair"] = f"{src}:{dst}"
                    row["frame"] = frame
                    rows.append(row)
                    print(f"  {row['pair']:6s} {frame}  filled "
                          f"{row['warp_psnr_filled']:.2f} -> {row['ours_psnr_filled']:.2f} dB")
        write_report(a.run, rows, cfg)
        return 0

    steps = [s for s in a.steps.split(",") if s]
    for (src, dst) in pairs:
        for fi in frames:
            frame = io_utils.frame_name(fi)
            fdir = os.path.join(run_root, f"cam{src}-cam{dst}-{frame}")
            os.makedirs(fdir, exist_ok=True)
            tag = f"cam{src}-cam{dst}-{frame}"
            print(f"\n===== {tag} =====")
            # EVERY STAGE READS ITS INPUT FROM `output/<run>/<stage>`, so a multi-frame run
            # that drives all frames with one run name lets frame n+1 overwrite the inputs of
            # frame n (measured: frames 5 and 6 ended up with a removal mask authored for a
            # different frame, which dropped their valid-pixel PSNR from 31.8 to 23.7 dB).
            # Each frame therefore gets its OWN run name; the stage dirs of a run are then
            # consistent by construction, and the per-frame copy below is what the report and
            # the analysis tools read.
            frame_run = f"{a.run}__cam{src}-{dst}-{frame}"
            args = ["--run", frame_run, "--dataset_root", cfg.dataset_root,
                    "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame]
            if a.config:
                args += ["--config", a.config]
            ok = True
            for s in steps:
                # only step6 owns --fill_oofa; every other stage would reject the flag
                sargs = list(args)
                if a.fill_oofa and s == "step6_render.py":
                    sargs += ["--fill_oofa"]
                t0 = time.time()
                rc, out, dt = run_step(s, sargs, log_path)
                npass = out.count("[CHECK][PASS]")
                nfail = out.count("[CHECK][FAIL]")
                print(f"  {s:24s} exit={rc} checks={npass} pass/{nfail} fail  {dt:5.1f}s")
                if rc != 0:
                    ok = False
                    tail = "\n".join(out.strip().splitlines()[-12:])
                    print(f"    !! {s} failed, tail:\n{tail}")
                    if not a.continue_on_fail:
                        break
                    print("    (--continue_on_fail: running the remaining stages anyway)")
            # keep a per-frame copy of the key artefacts so multi-frame runs stay readable
            # and so later analysis (ablation.py, summarize.py) never mixes frames
            for stage in ("calib", "preproc", "warp", "class", "removal", "fill", "final"):
                sdir = io_utils.run_dir(frame_run, stage, create=False)
                if os.path.isdir(sdir):
                    copy_tree(sdir, os.path.join(fdir, io_utils.STAGE_DIRS[stage]))
            row = evaluate(a.run, src, dst, fi, cfg, ablation=a.ablation,
                           frame_dir=fdir)
            row["pair"] = f"{src}:{dst}"
            row["frame"] = frame
            row["internal_checks_ok"] = bool(ok)
            rows.append(row)
            print("  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in row.items() if k in
                                  ("warp_psnr", "ghost_psnr", "direct_psnr", "ours_psnr",
                                   "ours_psnr_filled", "warp_psnr_filled",
                                   "warp_ssim", "ours_ssim")))

    write_report(a.run, rows, cfg)
    return 0


def copy_tree(src, dst):
    """Copy the small per-frame artefacts (png/npy/npz/json/txt), skip huge npz."""
    import shutil
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if not os.path.isfile(p):
            continue
        if os.path.getsize(p) > 24 * 1024 * 1024:
            continue
        shutil.copy2(p, os.path.join(dst, name))


def masked_psnr_ssim(pred, gt, mask, do_ssim=True):
    """PSNR/SSIM restricted to `mask` pixels.

    PSNR is the honest region-restricted value: MSE is averaged over the MASK pixels only
    (so it is comparable across arms and to the paper's hole-region numbers).  It is NOT
    computed after substituting GT outside the mask -- that would divide the same error
    sum by the whole frame and inflate the result by ~10*log10(N_total/N_mask) dB.

    SSIM needs local windows, so it is computed on the frame with GT substituted outside
    the region and then read out on the mask; that keeps every window fully defined.  The
    substitution inflates SSIM slightly (well-matched pixels enter the windows), which is
    why the SSIM column is reported as "region-window SSIM", indicative only.
    """
    a = np.array(pred, np.uint8)
    b = np.array(gt, np.uint8)
    m = np.asarray(mask, bool)
    if not m.any():
        return float("nan"), float("nan")
    p = metrics.psnr(a, b, mask=m)
    s = float("nan")
    if do_ssim:
        a2 = a.copy()
        a2[~m] = b[~m]
        s = metrics.ssim(a2, b, mask=m)
    return p, s


def evaluate(run, src, dst, fi, cfg, ablation=False, frame_dir=None):
    """Compare the pipeline artefacts against the real capture of camera `dst`.

    `frame_dir` selects a per-frame artefact copy (used by --eval_only); when omitted the
    stage directories of `run` are used directly.
    """
    out = {}
    fdir = os.path.join(frame_dir, "60_final") if frame_dir else \
        io_utils.run_dir(run, "final")
    wdir = os.path.join(frame_dir, "20_warp") if frame_dir else \
        io_utils.run_dir(run, "warp")
    frame = io_utils.frame_name(fi)
    gt = io_utils.load_view(cfg.dataset_root, dst, frame)["color"]

    final_path = os.path.join(fdir, "final.png")
    if not os.path.isfile(final_path):
        return out
    fin = io_utils.imread(final_path)
    warp = io_utils.imread(os.path.join(wdir, "warped_color.png"))
    hole = io_utils.imread(os.path.join(wdir, "hole_all.png"), gray=True) > 0
    disocc = io_utils.imread(os.path.join(wdir, "hole_disocc.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(wdir, "hole_oofa.png"), gray=True) > 0
    filled = hole & ~oofa                       # disocclusions + cracks: what we fill
    valid = ~hole                               # untouched by the warp: sanity baseline

    out["n_hole"] = int(hole.sum())
    out["n_disocc"] = int(disocc.sum())
    out["n_oofa"] = int(oofa.sum())
    out["hole_pct"] = 100.0 * float(hole.mean())

    # whole frame (what the paper's Table 1 numbers are; OOFA stays black there)
    out["ours_psnr"] = metrics.psnr(fin, gt)
    out["ours_ssim"] = metrics.ssim(fin, gt)
    out["warp_psnr"] = metrics.psnr(warp, gt)
    out["warp_ssim"] = metrics.ssim(warp, gt)
    # region-restricted, GT-substituted (the meaningful comparison)
    for key, m in (("valid", valid), ("filled", filled), ("disocc", disocc),
                   ("oofa", oofa)):
        o_p, o_s = masked_psnr_ssim(fin, gt, m, cfg.compute_ssim)
        w_p, w_s = masked_psnr_ssim(warp, gt, m, cfg.compute_ssim)
        out[f"ours_psnr_{key}"] = o_p
        out[f"ours_ssim_{key}"] = o_s
        out[f"warp_psnr_{key}"] = w_p
        out[f"warp_ssim_{key}"] = w_s
    out["psnr_gain_filled"] = out["ours_psnr_filled"] - out["warp_psnr_filled"]
    if ablation:
        p = os.path.join(wdir, "warped_color_noprep.png")
        if os.path.isfile(p):
            wn = io_utils.imread(p)
            wp, ws = masked_psnr_ssim(wn, gt, filled, cfg.compute_ssim)
            out["warp_raw_psnr_filled"] = wp
            out["ghost_gain_db"] = out["warp_psnr_filled"] - wp
            out["noprep_psnr_disocc"] = masked_psnr_ssim(wn, gt, disocc)[0]
        p = os.path.join(frame_dir or io_utils.run_dir(run), "ablation", "direct_inpaint.png")
        if os.path.isfile(p):
            dp = io_utils.imread(p)
            out["direct_psnr"] = masked_psnr_ssim(dp, gt, filled, cfg.compute_ssim)[0]
    return out


def write_report(run, rows, cfg):
    ed = io_utils.run_dir(run, "eval")
    good = [r for r in rows if r and "ours_psnr" in r]
    lines = [f"# Evaluation report — run `{run}`", ""]
    lines.append(f"config: src_cam={cfg.src_cam} dst_cam={cfg.dst_cam}  "
                 f"th={cfg.th} rounds={cfg.ghost_rounds} patch={cfg.patch_size} "
                 f"search={cfg.search_w}x{cfg.search_h} depth_tol={cfg.depth_tol}")
    lines.append("")
    if good:
        def m(k, r=good):
            v = [x[k] for x in r if k in x and np.isfinite(x[k])]
            return float(np.mean(v)) if v else float("nan")
        lines.append("## Per frame")
        lines.append("")
        lines.append("`filled` = disocclusion + crack pixels (the region this method is "
                     "responsible for); `valid` = pixels the plain warp already had. "
                     "Region PSNR averages the MSE over the region's own pixels (honest, "
                     "comparable across arms); region SSIM uses GT-substituted windows and "
                     "is indicative only.")
        lines.append("")
        lines.append("| pair | frame | hole % | whole PSNR ours/warp | **filled PSNR warp->ours** "
                     "| filled SSIM warp->ours | disocc PSNR warp->ours |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in good:
            flag = "" if r.get("internal_checks_ok", True) else " **(!)**"
            lines.append(
                f"| {r['pair']}{flag} | {r['frame']} | {r['hole_pct']:.2f} | "
                f"{r['ours_psnr']:.2f} / {r['warp_psnr']:.2f} | "
                f"{r['warp_psnr_filled']:.2f} -> **{r['ours_psnr_filled']:.2f}** | "
                f"{r['warp_ssim_filled']:.3f} -> **{r['ours_ssim_filled']:.3f}** | "
                f"{r['warp_psnr_disocc']:.2f} -> {r['ours_psnr_disocc']:.2f} |")
        lines += ["", "## Means", "",
                  f"- hole area                     : {m('hole_pct'):.2f} % "
                  f"({m('n_hole'):.0f} px, of which disocclusion {m('n_disocc'):.0f}, "
                  f"OOFA {m('n_oofa'):.0f})",
                  "- whole frame (includes black OOFA, as in the paper's tables):",
                  f"    - ours PSNR / SSIM          : {m('ours_psnr'):.3f} dB / "
                  f"{m('ours_ssim'):.4f}",
                  f"    - plain warp PSNR / SSIM    : {m('warp_psnr'):.3f} dB / "
                  f"{m('warp_ssim'):.4f}",
                  "- filled region (disocclusion + cracks):",
                  f"    - ours PSNR / SSIM          : {m('ours_psnr_filled'):.3f} dB / "
                  f"{m('ours_ssim_filled'):.4f}",
                  f"    - plain warp PSNR / SSIM    : {m('warp_psnr_filled'):.3f} dB / "
                  f"{m('warp_ssim_filled'):.4f}",
                  f"    - **gain**                  : "
                  f"{m('ours_psnr_filled') - m('warp_psnr_filled'):+.3f} dB / "
                  f"{m('ours_ssim_filled') - m('warp_ssim_filled'):+.4f}",
                  "- disocclusion only:",
                  f"    - ours PSNR                 : {m('ours_psnr_disocc'):.3f} dB "
                  f"(plain warp {m('warp_psnr_disocc'):.3f})",
                  "- pixels the warp already covered (regression check, must be ~unchanged):",
                  f"    - ours PSNR                 : {m('ours_psnr_valid'):.3f} dB "
                  f"(plain warp {m('warp_psnr_valid'):.3f})",
                  ]
        if "ghost_gain_db" in good[0]:
            lines.append(f"- ghost-removal gain (filled region, no-prep -> prep): "
                         f"{m('ghost_gain_db'):+.3f} dB")
        if "direct_psnr" in good[0]:
            lines.append(f"- direct virtual-view inpainting (baseline [17]) : "
                         f"{m('direct_psnr'):.3f} dB vs ours {m('ours_psnr_filled'):.3f} dB")
    else:
        lines.append("(no completed frames)")
    flagged = [r for r in rows if r and not r.get("internal_checks_ok", True)
               and "ours_psnr" in r]
    if flagged:
        lines += ["", "## Frames whose internal stage checks failed (metrics still measured)", "",
                  "These frames are scored anyway so the pair can be compared; the failing "
                  "invariant means the stage-4 removal/depth prediction is not trustworthy "
                  "there yet. See the stage checks.txt for the exact failed assertion.", ""]
        for r in flagged:
            lines.append(f"- {r['pair']} {r['frame']}: filled PSNR "
                         f"{r.get('warp_psnr_filled', float('nan')):.2f} -> "
                         f"{r.get('ours_psnr_filled', float('nan')):.2f} dB")
    if any(r.get("error") for r in rows):
        lines += ["", "## Failed frames", ""]
        for r in rows:
            if r.get("error"):
                lines.append(f"- {r.get('pair')} {r.get('frame')}")
    lines += ["", "## Artefacts", "",
              "- `../20_warp/panel_warp.png`, `panel_holetype.png`",
              "- `../30_class/panel_class.png`",
              "- `../40_removal/panel_removal.png`, `panel_depth_pred.png`",
              "- `../50_fill/panel_inpaint.png`",
              "- `../60_final/panel_final.png`"]
    io_utils.write_text(os.path.join(ed, "report.md"), "\n".join(lines) + "\n")
    io_utils.write_json(os.path.join(ed, "metrics.json"), rows)
    if good:
        chart = {}
        for r in good:
            chart.setdefault("warp", []).append(r["warp_psnr_filled"])
            chart.setdefault("ours", []).append(r["ours_psnr_filled"])
            if "noprep_psnr_disocc" in r:
                chart.setdefault("ghost", []).append(r["noprep_psnr_disocc"])
            if "direct_psnr" in r:
                chart.setdefault("inpaint_direct", []).append(r["direct_psnr"])
        metrics.progress_chart(chart, os.path.join(ed, "progress.png"),
                               keys=tuple(chart.keys()) or ("ours",),
                               title="PSNR on the filled region per frame")
    done = [r for r in rows if "ours_psnr" in r]
    print(f"\n[eval] {len(done)}/{len(rows)} frames evaluated -> {ed}\\report.md")
    for l in lines[:40]:
        print("  " + l)


if __name__ == "__main__":
    sys.exit(main())
