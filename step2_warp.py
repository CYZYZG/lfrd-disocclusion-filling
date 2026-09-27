"""Step 2 CLI -- modified 3D warping + crack handling + hole typing (paper III-B).

    python step2_warp.py --run ba54_f000
    python step2_warp.py --run fixture_synth --synthetic

Reads the preprocessed depth of step 1 (``<run>/10_preproc/depth_pp.png``) plus the
reference colour, forward-warps both with ``lfrd.warp.warp_view`` (sub-pixel 2x2 splat +
Z-buffer), fills the 1-2 px cracks, and splits the remaining holes into
crack / disocclusion / OOFA with ``lfrd.disocclusion.type_holes``.

Writes into ``output/<run>/20_warp/``:

    warped_color.png         virtual view (cracks filled)
    warped_color_noprep.png  ablation arm: warp with the RAW depth
    warped_depth.npy         int16 inverse depth, -1 = still-unfilled hole
                             (crack pixels are filled with the nearest valid depth)
    warped_lap.npy           warped Laplacian of the preprocessed depth (eq. 3 input)
    hole_all.png / hole_crack.png / hole_disocc.png / hole_oofa.png   (255 = hole)
    hole_type.png            uint8 0=valid 1=crack 2=disocc 3=oofa
    backward.npz             idx: flat source index per destination pixel (-1 = empty)
    warp_meta.npz            dx, dy, u_t, v_t, hole, src_oob (+ masks and counts)
    panel_warp.png           [reference | warped | noprep | hole | crack | oofa]
    panel_holetype.png       colour-coded typing overlay + close-up
    checks.txt

and ``output/<run>/panels/panel_holetype_zoom.png`` (close-up of the largest
disocclusion, crack=yellow, disocclusion=red, OOFA=blue).
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from lfrd import calib, cli, disocclusion, io_utils, reporter, viz, warp
from step1_preprocess import densest_window, load_synthetic_view, zoom_tile


# --------------------------------------------------------------------------- #
def fill_depth_nearest(depth, crack):
    """Fill ``crack`` pixels of a warped depth map (-1 = hole) from the nearest valid px.

    Same idea as ``warp.fill_cracks`` but the donor set is restricted to pixels that
    carry a defined depth (``depth >= 0``), so a crack is never filled from another hole.
    """
    out = np.array(depth, copy=True)
    crack = np.asarray(crack, bool)
    if not crack.any():
        return out
    valid = (out >= 0) & ~crack
    if not valid.any():
        return out
    iy, ix = warp.nearest_valid_index(valid)
    out[crack] = out[iy[crack], ix[crack]]
    return out


def aspect_box(box, shape, max_ratio=1.6):
    """Shrink a crop box until its aspect ratio is within ``max_ratio`` (for readable
    close-ups of long thin disocclusions)."""
    x0, y0, w, h = box
    H, W = shape
    if h > max_ratio * w:
        h = int(max_ratio * w)
        y0 = int(np.clip(y0 + (box[3] - h) // 2, 0, max(0, H - h)))
    if w > max_ratio * h:
        w = int(max_ratio * h)
        x0 = int(np.clip(x0 + (box[2] - w) // 2, 0, max(0, W - w)))
    return (x0, y0, w, h)


def brute_force_zbuf(dx, dy, z, box):
    """Explicit, slow reference implementation of the sub-pixel Z-buffer.

    For every destination pixel inside ``box`` all source pixels whose 2x2 sub-pixel
    splat covers it are collected and the nearest one wins (largest ``z``; ties are
    broken by the last splat pass, which is what ``warp.forward_warp`` does).

    Returns ``(win, zw)``: winning flat source index (-1 = no contribution) and the
    winning ``z`` (-inf where empty), both shaped (H, W).
    """
    dx = np.asarray(dx, np.float32)
    dy = np.asarray(dy, np.float32)
    z = np.asarray(z, np.float32)
    H, W = dx.shape
    x0, y0, w, h = box
    y, x = np.mgrid[0:H, 0:W]
    tx = x + dx
    ty = y + dy
    ix0 = np.floor(tx).astype(np.int32)
    iy0 = np.floor(ty).astype(np.int32)
    wx = (tx - ix0).astype(np.float32)
    wy = (ty - iy0).astype(np.float32)
    win = np.full((H, W), -1, np.int64)
    zw = np.full((H, W), -np.inf, np.float32)
    offsets = ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
               (0, 1, (1 - wx) * wy), (1, 1, wx * wy))
    for _order, (ox, oy, ww) in enumerate(offsets):
        txo = ix0 + ox
        tyo = iy0 + oy
        sel = (ww > 0) & (txo >= x0) & (txo < x0 + w) & (tyo >= y0) & (tyo < y0 + h)
        if not sel.any():
            continue
        src_flat = (y[sel] * W + x[sel]).astype(np.int64)
        zz = z[sel]
        tgt_flat = tyo[sel].astype(np.int64) * W + txo[sel].astype(np.int64)
        for k in range(tgt_flat.size):
            t = int(tgt_flat[k])
            zk = float(zz[k])
            # later passes overwrite ties, mirroring forward_warp's index assignment
            if zk > zw.flat[t] or (zk == zw.flat[t] and win.flat[t] >= 0):
                zw.flat[t] = zk
                win.flat[t] = int(src_flat[k])
    return win, zw


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = cli.base_parser("step 2: modified 3D warping + hole typing (paper III-B)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the synthetic fixture scene instead of the dataset view")
    ap.add_argument("--preproc_dir", default=None,
                    help="directory holding depth_pp.png (default <run>/10_preproc)")
    ap.add_argument("--hole_frac_min", type=float, default=0.02)
    ap.add_argument("--hole_frac_max", type=float, default=0.40)
    args = ap.parse_args(argv)
    cfg = cli.make_config(args)
    out = cli.stage_dir(args, "warp")
    pre_dir = args.preproc_dir or io_utils.run_dir(args.run, "preproc")
    cli.ensure_selection(args, cfg)
    rep = reporter.Reporter("step2", log_path=os.path.join(out, "checks.txt"))
    panels_dir = io_utils.run_dir(args.run, "panels")

    # ---- inputs --------------------------------------------------------- #
    if args.synthetic:
        cams = calib.load_calib(cfg.dataset_root, cfg.calib_name)
        ref = load_synthetic_view(cfg.src_cam)
    else:
        cams, ref = cli.run_context(args, cfg)
    color = ref["color"]
    depth_path = os.path.join(pre_dir, "depth_pp.png")
    if not os.path.isfile(depth_path):
        raise SystemExit(f"[step2] {depth_path} missing -- run step1_preprocess.py first")
    P_pp = io_utils.imread(depth_path, gray=True)
    if P_pp.shape != color.shape[:2]:
        raise SystemExit(f"[step2] depth_pp {P_pp.shape} does not match colour "
                         f"{color.shape[:2]} (run step1 for this --run/--synthetic pair)")
    P_raw = ref["depth"]
    rep.log(f"run={args.run} cam{cfg.src_cam}->cam{cfg.dst_cam} frame={ref.get('frame')} "
            f"synthetic={bool(args.synthetic)}")
    rep.log(f"depth_pp: {P_pp.shape} from {depth_path}; "
            f"changed vs raw = {int((P_pp != P_raw).sum())} px")

    # ---- warp (preprocessed depth = the paper's arm) -------------------- #
    wp = warp.warp_view(cams, cfg.src_cam, cfg.dst_cam, color, P_pp)
    hole = wp["hole"]
    crack_thin = warp.crack_mask(hole, cfg.crack_max_width)
    typing = disocclusion.type_holes(hole, crack_max_width=cfg.crack_max_width,
                                     min_area=cfg.holes_min_area,
                                     min_width=cfg.holes_min_width,
                                     src_oob=wp["src_oob"])
    warped_color = wp["warped_color"].copy()
    warped_depth_filled = wp["warped_depth"].copy()
    if cfg.crack_fill:
        warped_color = warp.fill_cracks(warped_color, crack_thin)
        warped_depth_filled = fill_depth_nearest(warped_depth_filled, crack_thin)
    depth_saved = np.rint(warped_depth_filled).astype(np.int16)
    still_hole = hole & ~crack_thin if cfg.crack_fill else hole

    # ---- ablation arm: raw (unpreprocessed) depth ----------------------- #
    wp_np = warp.warp_view(cams, cfg.src_cam, cfg.dst_cam, color, P_raw)
    crack_np = warp.crack_mask(wp_np["hole"], cfg.crack_max_width)
    color_np = wp_np["warped_color"].copy()
    if cfg.crack_fill:
        color_np = warp.fill_cracks(color_np, crack_np)
    typing_np = disocclusion.type_holes(wp_np["hole"], crack_max_width=cfg.crack_max_width,
                                        min_area=cfg.holes_min_area,
                                        min_width=cfg.holes_min_width)

    hf, hf_np = float(hole.mean()), float(wp_np["hole"].mean())
    rep.log(f"warp prep : hole {hf * 100:.4f}% ({int(hole.sum())} px)  "
            f"crack {typing['n_crack']}  disocc {typing['n_disocc']}  "
            f"oofa {typing['n_oofa']}  comps {typing['n_components']}")
    rep.log(f"warp noprep: hole {hf_np * 100:.4f}% ({int(wp_np['hole'].sum())} px)  "
            f"crack {typing_np['n_crack']}  disocc {typing_np['n_disocc']}  "
            f"oofa {typing_np['n_oofa']}")
    rep.log(f"thin crack mask (warp.crack_mask, max_width={cfg.crack_max_width}): "
            f"{int(crack_thin.sum())} px; typed crack = {typing['n_crack']} px "
            f"(equal by construction; enclosed remnants below min_area are still "
            f"disocc but flagged major=False)")
    rep.log(f"OOFA rule: {typing['oofa_method']} "
            f"(interior_gap_ok={typing['interior_gap_ok']})")
    for row in typing["table"][:5]:
        rep.log(f"  comp {row['label']:4d} area {row['area']:6d} "
                f"thick {row['max_thickness']:6.2f} bbox {row['bbox']} "
                f"{row['type_name']:6s} ({row['reason']}, major={row['major']}, "
                f"oob={row['oob_rate']:.2f})")

    # ---- artefacts ------------------------------------------------------ #
    io_utils.imwrite(os.path.join(out, "warped_color.png"), warped_color)
    io_utils.imwrite(os.path.join(out, "warped_color_noprep.png"), color_np)
    np.save(os.path.join(out, "warped_depth.npy"), depth_saved)
    np.save(os.path.join(out, "warped_lap.npy"), wp["warped_lap"].astype(np.float32))
    for name, m in (("hole_all", hole), ("hole_crack", typing["crack"]),
                    ("hole_disocc", typing["disocc"]), ("hole_oofa", typing["oofa"])):
        io_utils.imwrite(os.path.join(out, f"{name}.png"), (m * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(out, "hole_type.png"), typing["hole_type_map"])
    io_utils.save_npz(os.path.join(out, "backward.npz"),
                      idx=wp["backward"].astype(np.int32))
    io_utils.save_npz(
        os.path.join(out, "warp_meta.npz"),
        dx=wp["dx"].astype(np.float32), dy=wp["dy"].astype(np.float32),
        u_t=wp["u_t"].astype(np.float32), v_t=wp["v_t"].astype(np.float32),
        hole=hole, src_oob=wp["src_oob"], crack=typing["crack"],
        hole_type=typing["hole_type_map"], hole_noprep=wp_np["hole"],
        n_hole=np.int64(hole.sum()), n_crack=np.int64(typing["n_crack"]),
        n_disocc=np.int64(typing["n_disocc"]), n_oofa=np.int64(typing["n_oofa"]),
        hole_frac=np.float64(hf), hole_frac_noprep=np.float64(hf_np),
    )

    # ---- panels --------------------------------------------------------- #
    if not args.no_panel:
        ov_all = viz.mask_overlay(warped_color, hole, (255, 0, 0), 0.55, 0)
        ov_cr = viz.mask_overlay(warped_color, typing["crack"], (255, 255, 0), 0.8, 0)
        ov_of = viz.mask_overlay(warped_color, typing["oofa"], (0, 0, 255), 0.55, 0)
        viz.panel([
            (color, "reference colour"),
            (warped_color, "warped (prep depth)"),
            (color_np, "warped (raw depth, ablation)"),
            (ov_all, f"hole_all {hf * 100:.2f}%"),
            (ov_cr, f"crack (yellow) {typing['n_crack']} px"),
            (ov_of, f"OOFA (blue) {typing['n_oofa']} px"),
        ], path=os.path.join(out, "panel_warp.png"),
            title=f"step2 warp cam{cfg.src_cam}->cam{cfg.dst_cam} {ref.get('frame')}")

        bb = disocclusion.largest_bbox(typing["disocc"])
        if bb is not None:
            pad = 32
            x, y, w, h = bb
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            zoom_box = (x0, y0, min(hole.shape[1], x + w + pad) - x0,
                        min(hole.shape[0], y + h + pad) - y0)
        else:
            zoom_box = densest_window(hole, 128)
        zoom_box = aspect_box(zoom_box, hole.shape)
        ov_type = disocclusion.type_overlay(wp["warped_color"], typing)
        viz.panel([
            (wp["warped_color"], "warped, holes black"),
            (ov_type, "typing: crack=yellow disocc=red oofa=blue"),
            (disocclusion.type_color_map(typing), "hole_type map (0/1/2/3)"),
            (viz.mask_overlay(wp["warped_color"], typing["disocc"], (255, 0, 0), 0.55, 0),
             f"disocclusion (red) {typing['n_disocc']} px"),
            (zoom_tile(ov_type, zoom_box, 512, 384),
             f"close-up {zoom_box[2]}x{zoom_box[3]} @({zoom_box[0]},{zoom_box[1]})"),
        ], path=os.path.join(out, "panel_holetype.png"),
            title="step2 hole typing")
        viz.panel([(zoom_tile(ov_type, zoom_box, 640, 480), "typing close-up"),
                   (zoom_tile(wp["warped_color"], zoom_box, 640, 480),
                    "warped (holes black)"),
                   (zoom_tile(color, zoom_box, 640, 480), "reference colour")],
                  path=os.path.join(panels_dir, "panel_holetype_zoom.png"),
                  title=f"step2 hole typing close-up @({zoom_box[0]},{zoom_box[1]})")
        rep.log(f"zoom panel: {os.path.join(panels_dir, 'panel_holetype_zoom.png')} "
                f"crop {zoom_box[2]}x{zoom_box[3]} @({zoom_box[0]},{zoom_box[1]})")

    # ---- checks --------------------------------------------------------- #
    # 1. Z-buffer: brute force vs lfrd.warp.forward_warp on a 96x96 crop
    z_inv = (1.0 / np.maximum(calib.depth_from_P(P_pp), 1e-6)).astype(np.float32)
    win_h = max(96, int(0.15 * min(hole.shape)))
    box = densest_window(hole, win_h)
    cw, ch = min(96, hole.shape[1]), min(96, hole.shape[0])
    crop = (box[0], box[1], cw, ch)
    win_bf, z_bf = brute_force_zbuf(wp["dx"], wp["dy"], z_inv, crop)
    _c, _z, _h, _w, idx_core = warp.forward_warp(
        color.astype(np.float32), wp["dx"], wp["dy"], z=z_inv, rule="zbuf",
        splat="sub", return_index=True)
    x0, y0 = crop[0], crop[1]
    c_win = win_bf[y0:y0 + ch, x0:x0 + cw]
    i_win = idx_core[y0:y0 + ch, x0:x0 + cw]
    filled_bf, filled_core = c_win >= 0, i_win >= 0
    n_cov = int((filled_bf == filled_core).sum())
    both = filled_bf & filled_core
    n_both = int(both.sum())
    n_agree = int((c_win[both] == i_win[both]).sum())
    zc = z_inv.reshape(-1)
    tie_ok = True
    if n_both > n_agree:
        tie_ok = bool(np.allclose(zc[c_win[both]], zc[i_win[both]], atol=1e-6))
    rep.log(f"Z-buffer brute force on crop {crop}: coverage match {n_cov}/{cw * ch}, "
            f"winner identical {n_agree}/{n_both}, disagreements all z-ties: {tie_ok}")

    rep.check("hole fraction in sane range",
              args.hole_frac_min <= hf <= args.hole_frac_max and hole.any(),
              f"{hf * 100:.4f}% in [{args.hole_frac_min * 100:.1f}%,"
              f"{args.hole_frac_max * 100:.1f}%]")
    rep.check("backward idx >= 0 exactly where ~hole",
              bool(((wp["backward"] >= 0) == ~hole).all()),
              f"idx>=0 {int((wp['backward'] >= 0).sum())} vs valid {int((~hole).sum())}")
    rep.check("warped colour/depth agree on the hole set",
              bool(((wp["warped_depth"] < 0) == hole).all()),
              f"depth holes {int((wp['warped_depth'] < 0).sum())} vs colour "
              f"{int(hole.sum())}")
    rep.check("saved warped_depth.npy == -1 exactly on unfilled holes",
              bool(((depth_saved < 0) == still_hole).all()),
              f"{int((depth_saved < 0).sum())} unset px")
    rep.check("every hole pixel has exactly one type",
              bool((typing["crack"].astype(np.uint8) + typing["disocc"].astype(np.uint8)
                    + typing["oofa"].astype(np.uint8) == hole.astype(np.uint8)).all())
              and bool(hole.sum() == typing["n_crack"] + typing["n_disocc"]
                       + typing["n_oofa"]),
              f"crack {typing['n_crack']} + disocc {typing['n_disocc']} + "
              f"oofa {typing['n_oofa']} = {typing['n_hole']}")
    rep.check("crack mask (warp.crack_mask) subset of hole_all",
              bool((crack_thin & ~hole).sum() == 0),
              f"{int(crack_thin.sum())} px "
              f"({'present' if crack_thin.any() else 'none in this view'})")
    rep.check("thin cracks are always typed as cracks",
              bool((crack_thin & ~typing["crack"]).sum() == 0), "")
    rep.check("typed crack+disocc+oofa == hole_all",
              bool(((typing["crack"] | typing["disocc"] | typing["oofa"]) == hole).all()),
              "")
    rep.check("hole_type_map agrees with the masks",
              bool(((typing["hole_type_map"] == disocclusion.TYPE_CRACK) ==
                    typing["crack"]).all() and
                   ((typing["hole_type_map"] == disocclusion.TYPE_DISOCC) ==
                    typing["disocc"]).all() and
                   ((typing["hole_type_map"] == disocclusion.TYPE_OOFA) ==
                    typing["oofa"]).all()), "")
    rep.check("at least one major disocclusion component",
              typing["n_disocc_major"] > 0,
              f"{typing['n_disocc_major']} px (min_area={typing['min_area']}, "
              f"min_width={typing['min_width']})")
    oofa_rel = abs(typing["n_oofa"] - typing_np["n_oofa"]) / max(1.0, typing_np["n_oofa"])
    # The OOFA is the part of the virtual view the reference camera never captured, so it
    # must be present (nothing can fill it) and must not be grossly perturbed by the ghost
    # preprocessing.  The exact margin is geometry dependent: for cam5->cam4 the OOFA moves
    # by 1.27%, for cam5->cam6 (large baseline, where the shift is mostly vertical) 6.84%.
    # We therefore require the right ORDER OF MAGNITUDE, not a fixed percentage.
    rep.check("preprocessing leaves the OOFA region essentially unchanged",
              oofa_rel <= 0.15 and typing["n_oofa"] > 0,
              f"{typing['n_oofa']} vs noprep {typing_np['n_oofa']} px "
              f"({oofa_rel * 100:.2f}% change)")
    rep.check("crack filling only writes crack pixels",
              (not cfg.crack_fill) or
              bool((warped_color[~crack_thin] == wp["warped_color"][~crack_thin]).all()),
              f"{int(crack_thin.sum())} px filled")
    rep.check("warped colour is finite; raw warp holes stay black",
              bool(np.isfinite(wp["warped_color"]).all()) and
              (not hole.any() or int(wp["warped_color"][hole].max()) == 0),
              f"raw holes {int(hole.sum())} px, all zero: "
              f"{bool(not hole.any() or int(wp['warped_color'][hole].max()) == 0)}")
    rep.check("Z-buffer winner matches brute force on 96x96 crop",
              int((filled_bf == filled_core).sum()) == cw * ch and n_agree == n_both
              and tie_ok,
              f"coverage {n_cov}/{cw * ch}, winners {n_agree}/{n_both}, ties ok={tie_ok}")
    rep.finish()
    return 0


if __name__ == "__main__":
    main()
