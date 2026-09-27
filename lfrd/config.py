"""Run configuration: every tunable of the reproduction in one dataclass.

Paper values are the defaults; each field records its source in the docstring so that
deviations are always visible.  `RunConfig.save/load` round-trips through JSON so that
each run directory is self-describing.
"""
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Tuple

from . import io_utils


@dataclass
class RunConfig:
    # ---- data selection -------------------------------------------------- #
    run_name: str = "main"
    dataset_root: str = io_utils.DATASET_ROOT_DEFAULT
    calib_name: str = "ballet"
    src_cam: int = 5                 # reference camera (BA54 -> 5)
    dst_cam: int = 4                 # virtual camera   (BA54 -> 4)
    frames: Tuple[int, ...] = (0,)   # frame indices; use range n for batches

    # ---- 1. morphology-based depth preprocessing (paper III-A) ----------- #
    th: int = 20                     # paper IV-A: "th is set to 20"
    ghost_rounds: int = 2            # paper III-A: ghosts are 1-2 px wide -> twice
    ghost_min_change: int = 1        # ignore markings that change nothing

    # ---- 2. 3D warping (paper III-B) ------------------------------------- #
    splat: str = "sub"               # "sub" (2x2 sub-pixel) or "floor"/"round"
    rule: str = "zbuf"               # nearest layer wins (correct occlusion)
    crack_max_width: int = 2         # paper: cracks are 1-2 px wide
    crack_fill: bool = True          # "cracks are filled by the surrounding valid pixels"

    # ---- 3. disocclusion edge classification (paper III-B, eq. 3) -------- #
    use_laplacian: bool = True       # eq. (3) sign rule
    holes_min_area: int = 60         # ignore tiny holes when classifying
    holes_min_width: int = 4         # > this thickness -> disocclusion, else crack
    edge_band: int = 2               # width of the band used to sample edge depth

    # ---- 4. local foreground removal (paper III-C) ----------------------- #
    dilate_rm: int = 2               # paper: the extracted mask is morphologically dilated
    removal_from_bg_side: bool = True  # paper: "the foreground removal starts from the background side"
    partial_by_width: bool = True    # case 1: remove a region as wide as the disocclusion

    # ---- 5. removed-region filling (paper III-D) ------------------------- #
    patch_size: int = 9              # paper IV-A
    temporal_frames: int = 0         # 0 = off; N = use frames 0..N-1 as temporal background
    #   Paper IV/V future work.  With a static camera the occluded background a disocclusion
    #   exposes is usually visible in OTHER frames: the temporal percentile of the inverse
    #   depth is a background estimate and the frames reaching it carry the true colour.  Where
    #   a pixel has such evidence the invented occlusion-layer content is replaced by the real
    #   thing; pixels that never become background keep the single-view prediction.
    #   Measured on BA54 (frames 0-2, 100 frames): the occlusion layer scores 20.84 -> 22.74 dB
    #   (+1.90) and covers 39723 -> 41120 px.  ~30% of the removed pixels become background at
    #   some frame; the other ~70% never do and are unchanged.
    temporal_q: float = 10.0         # temporal percentile used as the background depth
    temporal_tol: float = 4.0        # a frame counts as background when within this tolerance
    temporal_agg: str = "mean"       # how the sampled frames are combined: "mean" (average),
    #   "median", or "consensus" (median, then average only the samples within
    #   temporal_agree_tol of it).  The frames a pixel samples differ from each other by ~8.4
    #   colour units, so aggregation was the obvious suspect for the "over-textured" blocks.
    #   Measured, it is NOT: on 5 frames (hole PSNR / SSIM) mean 22.668 / 0.6036,
    #   median 22.667 / 0.6055, consensus 21.645 / 0.5302.  Mean stays the default.
    temporal_agree_tol: float = 6.0  # agreement window for temporal_agg="consensus"
    refguide: bool = False           # reference-guided occlusion-layer prediction: search the
    #   REFERENCE IMAGE for the background the removed band will reveal, instead of inventing it
    #   (lfrd/refguide.py).  Off by default because it is only additive WITHOUT the temporal
    #   model -- with temporal on it overlaps and costs 0.67 dB.  Measured on BA54 frames 0-4
    #   through the real stage 6: paper-literal 21.00 -> 21.68 dB (+0.68) with refguide on;
    #   temporal 22.67 -> 22.46 (-0.21) restricted to the temporal gap and 22.00 (-0.67) applied
    #   everywhere.  Use it when no multi-frame sequence is available.
    refguide_search: int = 0         # displacement search half-window; 0 (the back-projected
    #   position only) measured best, and enlarging it made things worse, which confirms the
    #   geometry is right and only the CONTENT was missing
    refguide_slack: float = 8.0      # depth-consistency tolerance (8-bit inverse depth)
    refguide_cost: float = 48.0      # mean SSD/channel above which the match is rejected
    photo_correct: bool = True       # photometric seam match of the filled region, at the very
    #   end of stage 6.  tools/photometric_check.py found the sibling reproduction's per-block
    #   colour offset from the ground truth is far smaller than ours (7.28 vs 10.73 on smooth
    #   background, 7.12 vs 10.84 on textured) while its texture retention is LOWER -- so a large
    #   part of its lead is photometric, not content or geometry: our invented fill sits at a
    #   slightly wrong level, and a smooth region has nothing but its level to get wrong.  The
    #   correction matches the hole's per-channel mean and contrast to the surrounding VALID
    #   content at the seam (no ground truth involved).  Measured on BA54, 10 frames, all
    #   positive: paper-literal 21.18 -> 21.54 (+0.36), temporal 22.86 -> 23.31 (+0.45).
    photo_ring: int = 3              # width (px) of the boundary band used to estimate it
    photo_clip: float = 25.0         # cap on the level shift, in grey levels
    photo_contrast: bool = True      # also match the contrast, not only the mean
    sizes: Optional[Tuple[int, ...]] = None   # adaptive patch-size cascade -- EXPERIMENTAL,
    #   OFF by default because it measured WORSE on this data.  Set to (9, 7, 5, 3) to
    #   enable.  Numbers and reasoning in 提升空间分析.md §4:
    #     * cam6->cam7 (large baseline, mostly static background): the cascade beats the
    #       single 9x9 on 3/3 frames (+0.46 / +0.50 / +0.62 dB).
    #     * cam5->cam4 (BA54, moving dancers): it LOSES on 3/3 frames with the
    #       "accept the first size whose cost <= beta" rule, and the
    #       "lowest mean-SSD wins" rule collapses to 3x3 for 1626 of 1859 iterations and
    #       drops f000 from 19.57 to 18.95 dB.
    #   The failure mode is structural: a 3x3 patch finds a low-cost match almost anywhere, so
    #   any size-adaptive rule that trusts the match cost lets it pre-empt the 9x9 exactly
    #   where the larger patch would have carried the structure.  The sibling DIBR project gets
    #   away with it because its search window is 69 px (not 160x120) and its candidates are
    #   restricted to an eroded background band, so a 3x3 there cannot roam.
    size_rule: str = "best"          # "best" (lowest mean SSD/channel, `size_bonus` pref)
    #   or "cascade" (accept the first size with cost <= beta, largest first)
    size_bonus: float = 0.15         # larger-patch preference under size_rule="best"
    beta: float = 150.0              # acceptance threshold for size_rule="cascade"
    beta_mode: str = "mean"          # "mean" (gray-level units) | "sum" (literal SSD)
    struct_pen: float = 0.0          # cross-row structural penalty
    #   Penalises borrowing a source patch from a vertically displaced row:
    #   cost += struct_pen * w * dy^2 * (C * n_valid), with w = mean |vertical gradient| in a
    #   15x15 window (clamped to 3).  It helps a rectified 1-D disparity pipeline keep
    #   horizontal structures (rails, barres) aligned, but our matcher already has the eq. (9)
    #   depth-layer guard and the DD radius ladder doing that job, and every value tested
    #   here made the result WORSE (f000 disocclusion PSNR: 0.0 -> 22.94, 2.0 -> 22.82,
    #   8.0 -> 21.98, 30.0 -> 21.4).  Kept as a documented, switchable option; default off.
    search_w: int = 160              # paper IV-A
    search_h: int = 120              # paper IV-A
    depth_tol: float = 0.2           # eq. (9): DD <= 0.2
    alpha: int = 255                 # Criminisi data-term normalisation
    use_bg_term: bool = True         # eq. (8) B(p)
    use_depth_term: bool = True      # eq. (7) Z(p)
    use_depth_limit: bool = True     # eq. (9)-(10)
    local_search: bool = True        # global -> local search (paper III-D)
    max_iters: int = 200000
    save_priority_every: int = 0     # >0: dump priority snapshots (debug)

    # ---- 6. disocclusion filling & postprocessing (paper III-E) ---------- #
    fill_oofa: bool = False          # ambiguous in the paper; off by default
    postprocess: bool = True         # fill remaining small holes (B(p) == 1)

    # ---- 7. evaluation --------------------------------------------------- #
    compute_ssim: bool = True
    keep_intermediate: bool = True
    verbose: bool = True

    # ------------------------------------------------------------------ #
    def to_dict(self):
        d = asdict(self)
        d["frames"] = list(self.frames)
        return d

    def save(self, path):
        io_utils.write_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        if "frames" in d:
            d["frames"] = tuple(int(x) for x in d["frames"])
        return cls(**d)

    def save_to_run(self, run_name=None):
        rn = run_name or self.run_name
        path = os.path.join(io_utils.run_dir(rn), "config.json")
        self.save(path)
        return path


def parse_frames(spec, n_frames: int = 100):
    """'all' | '0' | '0,5,9' | '0-9' -> tuple of frame indices."""
    if isinstance(spec, (list, tuple)):
        return tuple(int(x) for x in spec)
    s = str(spec).strip().lower()
    if s in ("all", ""):
        return tuple(range(n_frames))
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(sorted(set(out)))
