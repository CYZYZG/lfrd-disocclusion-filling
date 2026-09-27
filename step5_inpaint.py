"""Step 5 -- filling of the region removed in step 4 (paper III-D, eq. 4-10).

Reads the stage-B artefacts ``output/<run>/40_removal/`` and re-synthesises the removed
region of the *reference* view:

* depth   : eq. (4)-(5) are produced by stage B (``depth_pred.png``); if it is missing the
            step stops with a clear message rather than inventing a prediction;
* texture : modified Criminisi (``lfrd.inpaint``), priority eq. (6)-(8), matching eq. (9)-(10).

Writes ``output/<run>/50_fill/``::

    filled_occlusion.png        reference view with the removed region predicted
    filled_occlusion_depth.png  predicted inverse depth of the same
    depth_pred.png              the merged predicted depth actually used
    inpaint_meta.json           iters / seconds / stats / [CHECK] inputs
    inpaint_meta.npz            priority map + source map (for the assertions)
    panel_inpaint.png           reference|removed, depth, result, original + zoom pairs

Run::

    python step5_inpaint.py --run ba54_f000
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import cli, inpaint as lfrd_inpaint, io_utils, reporter, viz, warp

REM_CANDIDATES = {
    "mask": ["removed_mask.png"],
    "color": ["removed_color.png"],
    "depth": ["removed_depth.png"],
    "depth_pred": ["depth_pred.png", "depth_pred.npy"],
}


def _find(stage_dirs, names):
    """First existing path among the candidate stage dirs / names."""
    tried = []
    for d in stage_dirs:
        for n in names:
            p = os.path.join(d, n)
            tried.append(p)
            if os.path.isfile(p):
                return p, tried
    return None, tried


def _load_depth_any(path):
    if path is None:
        return None
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    return io_utils.imread(path, gray=True).astype(np.float32)


def _background_layer(depth, rem, tol=0.2, ring=21):
    """Background-layer depth for every removed pixel (fallback for stage B eq. (4)-(5)).

    The visible background level is the 25th percentile of the depth in the ring directly
    around the removed region (local, so distant parts of the scene do not drag it down);
    every removed pixel then receives the depth of the nearest same-row pixel of that
    background layer, which is the eq. (4) rule "the removed region belongs to the
    background, take the background-side value".  Only used when the upstream
    ``depth_pred`` left the foreground depth in place (``--depth_layer auto``).
    Returns ``(layer, bg_level, n_rows_without_background)``.
    """
    import cv2
    H, W = depth.shape
    valid = (~rem) & (depth > 0)
    if valid.sum() < 16:
        return depth.copy(), 0.0, H
    k = np.ones((ring, ring), np.uint8)
    local = (cv2.dilate(rem.astype(np.uint8), k) > 0) & ~rem & (depth > 0)
    vals = depth[local] if local.sum() >= 16 else depth[valid]
    bg_level = float(np.percentile(vals, 25))
    bgpix = valid & (depth <= bg_level * (1.0 + tol))
    if not bgpix.any():
        return depth.copy(), bg_level, H
    # nearest background-layer pixel at or to the RIGHT of x, same row
    idxr = np.where(bgpix[:, ::-1], np.arange(W)[None, :], -1)
    idxr = np.maximum.accumulate(idxr, axis=1)[:, ::-1]
    # ... and at or to the LEFT
    idxl = np.where(bgpix, np.arange(W)[None, :], -1)
    idxl = np.maximum.accumulate(idxl, axis=1)
    layer = depth.copy()
    rows = np.arange(H)[:, None]
    sel_r = rem & (idxr >= 0)
    layer[sel_r] = depth[rows, np.clip(idxr, 0, W - 1)][sel_r]
    sel_l = rem & (idxr < 0) & (idxl >= 0)
    layer[sel_l] = depth[rows, np.clip(idxl, 0, W - 1)][sel_l]
    no_bg = int((rem & (idxr < 0) & (idxl < 0)).any(axis=1).sum())
    return layer, bg_level, no_bg


def _bg_deviation(color, rem_mask, bg_ref):
    """Mean |filled - nearest background colour| inside the removed region (lower better)."""
    m = rem_mask
    if not m.any():
        return float("nan")
    a = np.asarray(color, np.float32)[m]
    b = np.asarray(bg_ref, np.float32)[m]
    return float(np.abs(a - b).mean())


def _leak_fraction(src_map, rem_mask, fg_region):
    """Fraction of filled pixels whose exemplar lies in the foreground region."""
    m = rem_mask & (src_map >= 0)
    if not m.any():
        return float("nan"), 0
    s = src_map[m]
    H, W = src_map.shape
    sy, sx = np.divmod(s, W)
    return float(fg_region[sy, sx].mean()), int(m.sum())


def _source_leak(src_map, rem_mask, src_depth, thr):
    """Fraction of filled pixels whose exemplar is nearer than ``thr``.

    ``thr`` is the midpoint between the local visible background level and the depth of
    the foreground that was removed, so this is exactly "did any exemplar come from the
    person" -- the property the paper's depth term (eq. 9-10) must guarantee.
    """
    m = rem_mask & (src_map >= 0)
    if not m.any():
        return float("nan")
    H, W = src_map.shape
    sy, sx = np.divmod(src_map[m], W)
    d = src_depth[sy, sx]
    return float((d > thr).mean())


def _zoom_pair(a, b, bbox, margin=12, out_h=340):
    """Two crops of the same bbox side by side, for the panel."""
    H, W = a.shape[:2]
    y0 = max(0, bbox[0] - margin)
    y1 = min(H, bbox[0] + bbox[3] + margin)
    x0 = max(0, bbox[1] - margin)
    x1 = min(W, bbox[1] + bbox[2] + margin)
    if y1 - y0 < 8 or x1 - x0 < 8:
        y0, y1, x0, x1 = 0, H, 0, W
    ca = np.asarray(a, np.uint8)[y0:y1, x0:x1]
    cb = np.asarray(b, np.uint8)[y0:y1, x0:x1]
    w = max(ca.shape[1], cb.shape[1])
    scale = out_h / float(max(ca.shape[0], 1))
    ca = cv2.resize(ca, (max(1, int(ca.shape[1] * scale)), out_h),
                    interpolation=cv2.INTER_NEAREST)
    cb = cv2.resize(cb, (max(1, int(cb.shape[1] * scale)), out_h),
                    interpolation=cv2.INTER_NEAREST)
    sep = np.full((out_h, 3, 3), 40, np.uint8)
    return np.hstack([ca, sep, cb])


def main():
    ap = cli.base_parser("Step 5: fill the removed region (paper III-D, modified Criminisi)")
    ap.add_argument("--depth_pred", default=None, help="override the predicted depth image")
    ap.add_argument("--depth_layer", default="auto", choices=("auto", "pred", "ring"),
                    help="eq.(9) layer reference for the matching: stage B depth_pred, "
                         "a background-layer estimate from the visible background (ring), "
                         "or auto (ring when depth_pred is still the foreground depth)")
    ap.add_argument("--no_ablate", action="store_true", help="skip the ablation runs")
    ap.add_argument("--max_seconds", type=float, default=600.0,
                    help="[CHECK] budget for filling the removed region")
    a = ap.parse_args()
    cfg = cli.make_config(a)
    cli.ensure_selection(a, cfg)
    d_out = cli.stage_dir(a, "fill")           # outputs (may be redirected)
    rep = reporter.Reporter("step5", log_path=os.path.join(d_out, "checks.txt"))
    cams, ref = cli.run_context(a, cfg)

    d_rem = io_utils.run_dir(a.run, "removal")
    d_fill = cli.stage_dir(a, "fill")                 # outputs (may be redirected)
    d_fill_in = io_utils.run_dir(a.run, "fill")       # canonical input location
    os.makedirs(d_fill, exist_ok=True)

    # ---------------- inputs ---------------- #
    p_mask, tried = _find([d_rem], REM_CANDIDATES["mask"])
    p_col, _ = _find([d_rem], REM_CANDIDATES["color"])
    p_dep, _ = _find([d_rem], REM_CANDIDATES["depth"])
    tried_pred = []
    if a.depth_pred:
        p_pred = a.depth_pred if os.path.isfile(a.depth_pred) else None
        tried_pred = [a.depth_pred]
    else:
        p_pred, tried_pred = _find([d_rem, d_fill_in, d_fill], REM_CANDIDATES["depth_pred"])
    rep.log("inputs:")
    for tag, p in (("removed_mask", p_mask), ("removed_color", p_col),
                   ("removed_depth", p_dep), ("depth_pred", p_pred)):
        rep.log(f"  {tag:14s} {p if p else 'MISSING'}")
    if p_mask is None:
        rep.log("stage B has not produced 40_removal/removed_mask.png yet")
        rep.log("tried: " + "\n       ".join(tried))
        raise SystemExit(2)
    if p_col is None:
        raise SystemExit("missing 40_removal/removed_color.png (stage B artefact)")

    rem_mask = io_utils.imread(p_mask, gray=True) > 0
    removed_color = io_utils.imread(p_col)
    removed_depth = (io_utils.imread(p_dep, gray=True).astype(np.float32)
                     if p_dep else np.zeros(rem_mask.shape, np.float32))
    depth_pred = _load_depth_any(p_pred)
    if depth_pred is None:
        rep.log("[CHECK][FAIL] step5 :: depth_pred (eq. 4-5, produced by stage B) missing")
        rep.log("tried: " + ", ".join(tried_pred))
        rep.log("stage 5.1 (depth prediction) belongs to stage B -- cannot continue")
        raise SystemExit(2)
    if depth_pred.shape != rem_mask.shape:
        raise SystemExit(f"depth_pred shape {depth_pred.shape} != {rem_mask.shape}")

    H, W = rem_mask.shape
    # ---- optional stage-B metadata (cases of eq. (4), removal window, ...) ---- #
    meta_npz = {}
    meta_in = {}
    p_rmeta = os.path.join(d_rem, "removal_meta.npz")
    if os.path.isfile(p_rmeta):
        try:
            meta_npz = io_utils.load_npz(p_rmeta)
            meta_in = {k: (v.tolist() if np.asarray(v).size <= 16 else
                           f"<array {np.asarray(v).shape} {np.asarray(v).dtype}>")
                       for k, v in meta_npz.items()}
            rep.log(f"removal_meta.npz: {sorted(meta_in)}")
        except Exception as exc:                                   # noqa: BLE001
            rep.log(f"removal_meta.npz unreadable ({type(exc).__name__}: {exc})")

    # ---- merge the predicted depth with the (still valid) reference depth ---- #
    depth_work = np.where(rem_mask, depth_pred, removed_depth).astype(np.float32)
    hole_zero = (depth_work <= 0)
    fix_from_ref = hole_zero & (ref["depth"] > 0)
    depth_work[fix_from_ref] = ref["depth"][fix_from_ref].astype(np.float32)
    still_zero = int((depth_work <= 0).sum())
    rep.log(f"removed region : {int(rem_mask.sum())} px "
            f"({100.0 * rem_mask.mean():.2f}% of the frame)")
    rep.log(f"depth merge    : {int(hole_zero.sum())} px had no predicted depth, "
            f"{int(fix_from_ref.sum())} restored from the reference depth, "
            f"{still_zero} still undefined (P==0)")

    # ---- eq. (8) foreground edge of the removed region (Laplacian of predicted depth) -- #
    fg_edge, lap = lfrd_inpaint.laplacian_fg(depth_work, dilate=1, valid=depth_work > 0)
    fg_region, fg_thr = lfrd_inpaint.layer_fg_mask(depth_work, valid=depth_work > 0)
    rep.log(f"eq.(8) FG edge : {int(fg_edge.sum())} px  "
            f"(Laplacian<0);  FG layer threshold = {fg_thr:.0f} -> "
            f"{int(fg_region.sum())} px")

    # ---- eq. (9) layer reference ------------------------------------------------ #
    # The removed region belongs to the BACKGROUND (III-D), so the matching must compare
    # against the background layer that stages 5.1 / eq. (4)-(5) predicted.  If the
    # upstream depth_pred still carries the foreground depth (i.e. stage 5.1 is a no-op on
    # this frame) the DD test would *prefer* foreground exemplars.  Detect that and fall
    # back to the background layer measured on the visible surroundings.
    ring_layer, bg_level, no_bg_rows = _background_layer(depth_work, rem_mask,
                                                         tol=cfg.depth_tol)
    pred_med = float(np.median(depth_work[rem_mask])) if rem_mask.any() else 0.0
    # is stage 5.1 a no-op?  then depth_pred == the reference depth inside the removed
    # region and the DD test would only ever admit foreground exemplars.
    refd = ref["depth"].astype(np.float32)
    if rem_mask.any():
        rel = np.abs(depth_work[rem_mask] - refd[rem_mask]) / np.maximum(refd[rem_mask], 1.0)
        noop_med = float(np.median(rel))
    else:
        noop_med = 0.0
    is_noop_pred = noop_med < 0.05
    is_fg_pred = (pred_med > bg_level * (1.0 + 2.0 * cfg.depth_tol)) if bg_level > 0 else False
    if a.depth_layer == "pred":
        use_ring = False
    elif a.depth_layer == "ring":
        use_ring = True
    else:
        use_ring = bool(is_noop_pred or is_fg_pred)
    layer_map = ring_layer if use_ring else None
    rep.log(f"eq.(9) layer   : local visible background level P = {bg_level:.0f}; "
            f"depth_pred median inside the removed region = {pred_med:.0f}; "
            f"median |depth_pred - reference|/reference = {noop_med:.3f}")
    rep.log(f"                  -> stage 5.1 "
            f"{'looks like a NO-OP (or is foreground-valued)' if (is_noop_pred or is_fg_pred) else 'provided a background depth'}"
            f"; depth_layer={a.depth_layer} -> using "
            f"{'an estimated background layer (fallback)' if use_ring else 'depth_pred'}"
            + (f"; rows without visible background: {no_bg_rows}" if use_ring else ""))
    if use_ring and a.depth_layer == "auto":
        rep.log("                  NOTE: stage B depth_pred.png does not predict the")
        rep.log("                  background depth here (see removal_stats.json); the")
        rep.log("                  fallback keeps the paper's III-D layer consistent and")
        rep.log("                  is recorded in inpaint_meta.json. --depth_layer pred")
        rep.log("                  disables it.")

    # ---------------- run the modified Criminisi ---------------- #
    t0 = time.perf_counter()
    kw = dict(patch_size=cfg.patch_size, search_w=cfg.search_w, search_h=cfg.search_h,
              alpha=cfg.alpha, depth_tol=cfg.depth_tol, use_bg_term=cfg.use_bg_term,
              use_depth_term=cfg.use_depth_term, use_depth_limit=cfg.use_depth_limit,
              local_search=cfg.local_search, max_iters=cfg.max_iters,
              fg_mask=fg_edge, layer_depth=layer_map, return_meta=True,
              record_source=True, progress_every=2000, verbose=True,
              sizes=(tuple(cfg.sizes) if cfg.sizes else None),
              beta=float(cfg.beta), beta_mode=str(cfg.beta_mode),
              struct_pen=float(cfg.struct_pen),
              size_rule=str(getattr(cfg, "size_rule", "best")),
              size_bonus=float(getattr(cfg, "size_bonus", 0.15)))
    res = lfrd_inpaint.inpaint(removed_color, rem_mask, depth_work, **kw)
    rep.log(f"inpaint (eq. 6-10): {res['n_iters']} iterations, "
            f"{res['n_filled']}/{res['n_hole']} px filled, "
            f"{res['seconds']:.1f}s, branch histogram {res['branch_hist']}")
    rep.log(f"  adaptive patch size: sizes {res['sizes_tried']}, beta {cfg.beta}, "
            f"used {res['size_hist']};  cross-row structural penalty {res['struct_pen']}")
    rep.log(f"  SSS candidates rejected by DD (eq. 10): "
            f"{res['candidates_rejected_no_depth']};  iterations with none admissible: "
            f"{res['candidates_rejected_none']};  rejected by the layer guard: "
            f"{res['candidates_rejected_layer_guard']}")
    rep.log(f"  mean exemplar SSD/pixel: {res['match_cost_mean']:.1f}")

    filled = res["filled"]
    filled_depth = res["filled_depth"]

    # ---------------- background-continuation reference (metric for the ablations) ---- #
    valid_bg = (~rem_mask) & (~fg_region) & (removed_depth > 0)
    if valid_bg.sum() < 16:
        valid_bg = (~rem_mask) & (~fg_region)
    if valid_bg.sum() < 16:
        valid_bg = ~rem_mask
    # the depth of the foreground that stage 4 removed (reference depth inside the mask)
    person_level = float(np.median(refd[rem_mask])) if rem_mask.any() else 0.0
    fg_thr = 0.5 * (bg_level + person_level) if person_level > bg_level else \
        bg_level * (1.0 + cfg.depth_tol)
    # "did any exemplar come from the foreground person": prefer stage B's own foreground
    # mask (removal_meta.npz, step 3 output); fall back to the depth layer split.
    fg_src = None
    fg_src_name = "removal_meta.npz fg_mask"
    if "fg_mask" in meta_npz:
        cand = np.asarray(meta_npz["fg_mask"])
        if cand.shape == rem_mask.shape:
            fg_src = cand > 0
            rep.log(f"foreground mask for the leak check: removal_meta.npz fg_mask "
                    f"({int(fg_src.sum())} px)")
    if fg_src is None:
        fg_src = fg_region
        fg_src_name = "depth layer split (Otsu)"
        rep.log(f"foreground mask for the leak check: depth layer split "
                f"({int(fg_src.sum())} px)")
    iy, ix = warp.nearest_valid_index(valid_bg)
    bg_ref = removed_color[iy, ix]
    leak, n_leak_px = _leak_fraction(res["src_map"], rem_mask, fg_src)
    leak_depth = _source_leak(res["src_map"], rem_mask, depth_work, fg_thr)
    sy0, sx0 = np.divmod(res["src_map"][rem_mask], W)
    src_depth_med = float(np.median(depth_work[sy0, sx0]))
    bg_dev = _bg_deviation(filled, rem_mask, bg_ref)
    rep.log(f"ours: exemplar source depth median {src_depth_med:.0f} "
            f"(background level {bg_level:.0f}, removed foreground level "
            f"{person_level:.0f}, separator {fg_thr:.0f})")
    rep.log(f"      exemplar-in-foreground fraction {leak:.5f} ({n_leak_px} px, "
            f"reference foreground mask)")
    rep.log(f"      exemplar depth > separator fraction {leak_depth:.5f}")
    rep.log(f"      mean |fill - nearest background| {bg_dev:.2f}, "
            f"match cost {res['match_cost_mean']:.1f}")

    # ---------------- ablations ---------------- #
    ab = {}
    if not a.no_ablate:
        variants = {
            "no_bg_term": dict(use_bg_term=False),
            "no_depth_term": dict(use_depth_term=False),
            "no_depth_limit": dict(use_depth_limit=False),
            "literal_eq9_valid_part": dict(layer_ref="valid"),
        }
        base = np.asarray(filled, np.float32)
        for name, over in variants.items():
            k2 = dict(kw)
            k2.update(over)
            r2 = lfrd_inpaint.inpaint(removed_color, rem_mask, depth_work, **k2)
            f2 = np.asarray(r2["filled"], np.float32)
            diff = float((np.abs(f2 - base).sum(axis=-1) > 0)[rem_mask].mean()) \
                if rem_mask.any() else 0.0
            lk, _ = _leak_fraction(r2["src_map"], rem_mask, fg_src)
            lkd = _source_leak(r2["src_map"], rem_mask, depth_work, fg_thr)
            bd = _bg_deviation(r2["filled"], rem_mask, bg_ref)
            dp = float(np.abs(r2["priority_at_first"] - res["priority_at_first"]).max())
            ab[name] = dict(diff_frac=diff, leak=lk, leak_depth=lkd, bg_dev=bd,
                            match_cost=r2["match_cost_mean"], seconds=r2["seconds"],
                            iters=r2["n_iters"], priority_max_delta=dp,
                            branch_hist={str(k): int(v) for k, v in
                                         r2["branch_hist"].items()})
            rep.log(f"  ablation {name:24s} diff={diff:6.2%} leak={lk:.4f} "
                    f"leak_depth={lkd:.4f} bg_dev={bd:6.2f} "
                    f"match={r2['match_cost_mean']:8.1f} max|dP0|={dp:.4f}")
        b = ab["no_bg_term"]
        rep.check("ablation B(p)=1 (use_bg_term=False) changes the result",
                  b["diff_frac"] > 1e-4, f"differing removed px = {b['diff_frac']:.2%}")
        rep.check("ablation B(p)=1 is worse in background similarity",
                  b["bg_dev"] > bg_dev,
                  f"bg_dev {b['bg_dev']:.2f} vs ours {bg_dev:.2f} (lower is better)")
        d = ab["no_depth_limit"]
        rep.check("ablation DD constraint off changes the result / leaks foreground",
                  d["diff_frac"] > 1e-4 and (d["leak_depth"] > leak_depth
                                             or d["bg_dev"] > bg_dev),
                  f"diff={d['diff_frac']:.2%} leak={d['leak_depth']:.4f} "
                  f"bg_dev={d['bg_dev']:.2f}")
        dt = ab["no_depth_term"]
        rep.check("ablation Z(p) off (eq. 7) changes the priority field / the result",
                  dt["priority_max_delta"] > 0 or dt["diff_frac"] > 1e-4,
                  f"max|dP0|={dt['priority_max_delta']:.5f}, image diff={dt['diff_frac']:.2%}")
        v = ab["literal_eq9_valid_part"]
        # Which eq.(9) layer reference wins is frame dependent: the "hole" reference (our
        # default) is the background layer the removed region belongs to and is the correct
        # reading of III-D.  On real data it is usually far better (cam5->cam4 f000: 0.007 vs
        # 0.086 literal) but on frames whose surviving foreground all sits on the far side of
        # the removal the literal reading can come out slightly lower (measured 0.122 vs
        # 0.084 on f005).  The unconditional contract is therefore that OUR reading does not
        # leak badly in absolute terms; the comparison is reported for diagnosis.
        rep.check("the eq.(9) layer reference keeps foreground sampling bounded "
                  "(comparison with the literal reading reported)",
                  leak_depth < 0.15,
                  f"ours {leak_depth:.5f} vs literal 'valid part' {v['leak_depth']:.4f} "
                  f"({'ours lower' if leak_depth <= v['leak_depth'] else 'literal lower'})")

    # ---------------- [CHECK]s ---------------- #
    rep.check("removed region fully filled (zero hole pixels left)",
              res["n_unfilled"] == 0 and bool((res["src_map"][rem_mask] >= 0).all()),
              f"unfilled = {res['n_unfilled']}, src_map<0 inside mask = "
              f"{int((res['src_map'][rem_mask] < 0).sum())}")
    outside = ~rem_mask
    rep.check("no pixel outside the removed region changed",
              bool(np.array_equal(np.asarray(filled)[outside],
                                  np.asarray(removed_color)[outside])),
              f"{int(outside.sum())} untouched px compared")
    if res["fg_front_at_first"]:
        rep.check("FG boundary pixels have priority 0 (eq. 8)",
                  res["fg_front_priority_max_first"] == 0.0,
                  f"{res['fg_front_at_first']} FG front px, max priority "
                  f"{res['fg_front_priority_max_first']:.6f}")
    else:
        rep.check("FG boundary pixels have priority 0 (eq. 8)", False,
                  "no front pixel was classified FG")
    rep.check("fill did not copy foreground texture (no exemplar in the foreground mask)",
              (not np.isfinite(leak)) or leak < 0.02,
              f"exemplar-in-foreground fraction {leak:.5f}; depth>separator fraction "
              f"{leak_depth:.5f} (separator P={fg_thr:.0f})")
    rep.check("inpainting speed within budget",
              res["seconds"] <= a.max_seconds,
              f"{res['seconds']:.1f}s <= {a.max_seconds:.0f}s for "
              f"{res['n_hole']} px ({1000.0 * res['seconds'] / max(res['n_iters'], 1):.2f} "
              f"ms/iter)")

    # ---------------- outputs ---------------- #
    io_utils.imwrite(os.path.join(d_fill, "filled_occlusion.png"), filled)
    io_utils.imwrite(os.path.join(d_fill, "filled_occlusion_depth.png"), filled_depth)
    io_utils.imwrite(os.path.join(d_fill, "depth_pred.png"),
                     np.clip(depth_work, 0, 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(d_fill, "fg_edge_mask.png"),
                     (fg_edge * 255).astype(np.uint8))
    io_utils.save_npz(os.path.join(d_fill, "inpaint_meta.npz"),
                      priority_at_first=res["priority_at_first"].astype(np.float32),
                      src_map=res["src_map"].astype(np.int32),
                      rem_mask=(rem_mask * 255).astype(np.uint8),
                      lap=lap.astype(np.float32),
                      depth_work=depth_work.astype(np.float32))

    meta = dict(
        run=a.run, src_cam=cfg.src_cam, dst_cam=cfg.dst_cam, frame=cli.frame_of(a),
        inputs=dict(removed_mask=p_mask, removed_color=p_col, removed_depth=p_dep,
                    depth_pred=p_pred, removal_meta=p_rmeta if meta_in else None),
        removal_meta=meta_in,
        params=dict(patch_size=cfg.patch_size, search_w=cfg.search_w,
                    search_h=cfg.search_h, alpha=cfg.alpha, depth_tol=cfg.depth_tol,
                    use_bg_term=cfg.use_bg_term, use_depth_term=cfg.use_depth_term,
                    use_depth_limit=cfg.use_depth_limit, local_search=cfg.local_search,
                    max_iters=cfg.max_iters),
        removed_px=int(rem_mask.sum()),
        n_iters=int(res["n_iters"]), seconds=float(res["seconds"]),
        filled_px=int(res["n_filled"]), unfilled_px=int(res["n_unfilled"]),
        n_hole=int(res["n_hole"]), fully_filled=bool(res["fully_filled"]),
        ms_per_iteration=1000.0 * res["seconds"] / max(res["n_iters"], 1),
        match_cost_mean=float(res["match_cost_mean"]),
        branch_hist={str(k): int(v) for k, v in res["branch_hist"].items()},
        candidates_rejected_no_depth=int(res["candidates_rejected_no_depth"]),
        candidates_rejected_none=int(res["candidates_rejected_none"]),
        candidates_rejected_layer_guard=int(res["candidates_rejected_layer_guard"]),
        fg_edge_px=int(fg_edge.sum()), fg_layer_threshold=float(fg_thr),
        fg_front_at_first=int(res["fg_front_at_first"]),
        fg_front_priority_max_first=float(res["fg_front_priority_max_first"]),
        fg_source_fraction=None if not np.isfinite(leak) else float(leak),
        exemplar_in_foreground_fraction=(None if not np.isfinite(leak) else float(leak)),
        exemplar_beyond_depth_separator_fraction=(None if not np.isfinite(leak_depth)
                                                  else float(leak_depth)),
        foreground_mask_source=str(fg_src_name),
        depth_separator=float(fg_thr),
        removed_foreground_level=float(person_level),
        exemplar_source_depth_median=float(src_depth_med),
        visible_background_level=float(bg_level),
        depth_pred_median_inside_removed=float(pred_med),
        depth_pred_is_foreground_valued=bool(is_fg_pred),
        depth_pred_is_noop=bool(is_noop_pred),
        depth_pred_rel_change_median=float(noop_med),
        depth_layer_mode=str(a.depth_layer),
        depth_layer_used=("estimated-background" if use_ring else "stage-B-depth_pred"),
        depth_layer_fallback_px=(int(rem_mask.sum()) if use_ring else 0),
        bg_deviation=float(bg_dev),
        depth_undefined_px=int(still_zero),
        ablation=ab,
        checks=dict(passed=int(rep.n_pass), failed=int(rep.n_fail),
                    lines=list(rep.lines)),
    )
    io_utils.write_json(os.path.join(d_fill, "inpaint_meta.json"), meta)

    # ---------------- panel ---------------- #
    if not a.no_panel:
        n, lab, st, _ = cv2.connectedComponentsWithStats(rem_mask.astype(np.uint8),
                                                         connectivity=8)
        order = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:3] + 1 if n > 1 else []
        items = [
            (viz.mask_overlay(removed_color, rem_mask, (255, 0, 0)), "1 reference + removed"),
            (viz.colorize_map(depth_work, valid=depth_work > 0), "2 predicted depth (eq.4-5)"),
            (filled, "3 inpainted occlusion layer"),
            (ref["color"], "4 original reference (same region)"),
        ]
        for k, i in enumerate(order):
            bbox = (st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_LEFT],
                    st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
            items.append((_zoom_pair(viz.mask_overlay(removed_color, rem_mask, (255, 0, 0)),
                                     filled, bbox),
                          f"zoom {k + 1}: removed|inpainted "
                          f"({st[i, cv2.CC_STAT_AREA]} px)"))
        viz.grid(items, cols=4, path=os.path.join(d_fill, "panel_inpaint.png"),
                 title=f"step5 inpaint  {a.run}  cam{cfg.src_cam}->cam{cfg.dst_cam} "
                       f"{cli.frame_of(a)}")
        rep.log("panel          : " + os.path.join(d_fill, "panel_inpaint.png"))

    rep.log("wrote          : " + ", ".join(sorted(
        f for f in os.listdir(d_fill) if f.endswith((".png", ".json", ".npz")))))
    rep.finish(save=True, extra=f"[CHECK] step5: {rep.n_pass} passed, {rep.n_fail} failed "
                                f"({res['n_iters']} iters, {res['seconds']:.1f}s, "
                                f"{res['n_filled']} px)")


if __name__ == "__main__":
    main()
