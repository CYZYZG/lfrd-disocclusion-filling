"""Public interface: fill the disocclusions of a virtual view in ONE call.

    from lfrd.api import fill_holes, FillResult
    res = fill_holes("ba54", src_cam=5, dst_cam=4, frame="f000")
    res.save("out.png")

This is the whole validated pipeline (the paper's six stages plus the three measured
improvements) behind a single function:

  stage 1  morphology depth preprocessing (ghost removal)          paper III-A
  stage 2  forward 3D warping, crack handling, hole typing          paper II
  stage 3  disocclusion edge FG/BG classification                   paper III-B
  stage 4  local foreground removal + background depth prediction   paper III-C/D
  stage 5  modified-Criminisi occlusion-layer prediction            paper III-D
             + optional temporal background substitution           (measured +1.68 dB)
             + optional reference-guided fill                      (+0.68 dB, no sequence)
  stage 6  occlusion-layer compositing + postprocessing             paper III-E
             + optional OOFA filling                               (whole frame +9.5 dB)
             + photometric seam match (global + spatial drift)     (measured +0.95 dB)

Everything is classical image processing: numpy / opencv / scipy only, no learned model
(verify with tools/verify_no_dl.py).

Inputs
------
Either give the reference view explicitly (``ref_image`` / ``ref_depth`` arrays or paths), or
let the function load it from the MSR 3D Video Ballet dataset with ``dataset_root``.  Every
in-between artefact is written under ``output/<run>/`` exactly as the command-line pipeline
does, so the panels and checks.txt files are available for inspection.

Returns a :class:`FillResult`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import calib, io_utils

__all__ = ["fill_holes", "FillResult", "DEFAULT_STEPS"]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STEPS = ("step1_preprocess.py", "step2_warp.py", "step3_classify.py",
                 "step4_removal.py", "step5_inpaint.py", "step6_render.py")


def _copy_small(src: str, dst: str) -> None:
    """Copy the small per-frame artefacts, skipping anything above 24 MB."""
    import shutil
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if os.path.isfile(p) and os.path.getsize(p) <= 24 * 1024 * 1024:
            shutil.copy2(p, os.path.join(dst, name))


@dataclass
class FillResult:
    """Everything one call produced.

    Attributes
    ----------
    color : (H,W,3) uint8      the filled virtual view
    depth : (H,W) float32      its inverse depth (-1 where unknown)
    hole : (H,W) bool          pixels that were empty in the plain warp
    disocclusion, crack, oofa : (H,W) bool   the three hole types, as classified by stage 2
    filled_from_layer : (H,W) bool   pixels taken from the warped occlusion layer (stage 6)
    valid_untouched : (H,W) bool     pixels the plain warp already had (never modified)
    metrics : dict             PSNR/SSIM for the regions, when a ground-truth view exists
    run_dir : str              where all artefacts were written
    log : list[str]            the pipeline's own log lines
    """

    color: np.ndarray
    depth: np.ndarray
    hole: np.ndarray
    disocclusion: np.ndarray
    crack: np.ndarray
    oofa: np.ndarray
    filled_from_layer: np.ndarray
    valid_untouched: np.ndarray
    metrics: dict = field(default_factory=dict)
    run_dir: str = ""
    frame_dir: str = ""
    log: list = field(default_factory=list)
    stages: list = field(default_factory=list)

    # -- convenience ---------------------------------------------------------- #
    def save(self, path: str) -> str:
        """Write the filled view as a PNG (handles non-ASCII paths)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        io_utils.imwrite(path, self.color)
        return path

    def save_all(self, directory: str) -> dict:
        """Write the result plus the masks and the depth as a small bundle."""
        os.makedirs(directory, exist_ok=True)
        out = {}
        for name, img in (("filled.png", self.color),
                          ("hole.png", (self.hole * 255).astype(np.uint8)),
                          ("disocclusion.png", (self.disocclusion * 255).astype(np.uint8)),
                          ("crack.png", (self.crack * 255).astype(np.uint8)),
                          ("oofa.png", (self.oofa * 255).astype(np.uint8)),
                          ("filled_from_layer.png",
                           (self.filled_from_layer * 255).astype(np.uint8)),
                          ("valid_untouched.png",
                           (self.valid_untouched * 255).astype(np.uint8))):
            p = os.path.join(directory, name)
            io_utils.imwrite(p, img)
            out[name] = p
        io_utils.save_npz(os.path.join(directory, "filled_depth.npz"), depth=self.depth)
        out["filled_depth.npz"] = os.path.join(directory, "filled_depth.npz")
        return out

    def summary(self) -> str:
        """One-line human-readable summary."""
        h, w = self.color.shape[:2]
        n = int(self.hole.sum())
        filled = int((self.hole & ~self.oofa).sum())
        s = (f"{w}x{h} virtual view | hole {n} px ({100.0 * n / self.hole.size:.2f}%) | "
             f"filled {filled} px | OOFA {int(self.oofa.sum())} px")
        if self.metrics:
            m = self.metrics
            s += (f" | hole PSNR {m.get('psnr_filled', float('nan')):.2f} dB, "
                  f"whole {m.get('psnr_whole', float('nan')):.2f} dB")
        return s


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
def fill_holes(run: str,
               src_cam: int,
               dst_cam: int,
               frame: str = "f000",
               *,
               ref_image=None,
               ref_depth=None,
               calib_file: Optional[str] = None,
               dataset_root: Optional[str] = None,
               temporal_frames: int = 0,
               fill_oofa: bool = False,
               photo_correct: bool = True,
               refguide: bool = False,
               gt_cam: Optional[int] = None,
               gt_image=None,
               config=None,
               python: Optional[str] = None,
               quiet: bool = True,
               strict: bool = True,
               steps=DEFAULT_STEPS) -> FillResult:
    """Fill the disocclusions of virtual view ``dst_cam`` from reference view ``src_cam``.

    Two ways to feed it
    -------------------
    **A. MSR 3D Video Ballet** (the validated path, needs no extra arguments)::

        fill_holes("demo", src_cam=5, dst_cam=4, frame="f000")

    the reference image/depth, the calibration and the ground-truth view are all found under
    ``dataset_root`` (which defaults to the configured dataset location).

    **B. Your own pictures.**  View synthesis needs camera parameters, so you must supply them::

        fill_holes("mine", src_cam=0, dst_cam=1, frame="f000",
                   ref_image="left.png", ref_depth="left_depth.png",
                   calib_file="calibParams-mine.txt")

    The calibration file uses the MSR format, which is what :func:`lfrd.calib.parse_calib`
    reads:

        <name>            e.g. mine
        <n_cams>
        <K> 3x3 (fx, 0, cx / 0, fy, cy / 0, 0, 1), one block per camera
        <C> camera centre (X Y Z)
        <R> 3x3 rotation

    Only the two cameras named by ``src_cam``/``dst_cam`` are used.  Your depth must use the
    same 8-bit inverse-depth convention as the dataset (see 接口使用说明.md section 3) --
    that convention, not the picture size, is what the geometric stages assume.

    Parameters
    ----------
    run : str
        Name of the run; all artefacts go to ``output/<run>/``.  Reuse a name to overwrite.
    src_cam, dst_cam : int
        Reference and virtual camera indices as they appear in the calibration file.
    frame : str
        Frame name, e.g. ``"f000"``.  With ``temporal_frames > 0`` the same index is used to
        locate the target within the sequence.
    ref_image, ref_depth : (H,W,3)/(H,W) uint8 | str | None
        Your own reference view (array or path).  Give both or neither.
    calib_file : str | None
        Calibration file in the MSR format; required together with ``ref_image``.
    dataset_root : str | None
        Dataset root; defaults to the configured MSR Ballet location.
    temporal_frames : int
        **0 = single view** (the paper's setting).  ``N > 0`` additionally builds a temporal
        background model from frames ``0..N-1`` of the sequence and uses the real content
        wherever a hidden pixel becomes visible in some frame.  Measured +1.68 dB on-hole.
        Needs the whole sequence on disk, so it is not available with your own pictures.
    fill_oofa : bool
        Also fill the areas outside every source camera's field of view (the black borders).
        Worth ~+9.5 dB on the whole frame; the method's own region is unaffected.
    photo_correct : bool
        Two-stage photometric seam match of the filled region (default on, +0.95 dB).
    refguide : bool
        Depth-constrained patch search in the reference image.  Only useful WITHOUT the
        temporal model (+0.68 dB); with it the two overlap and it costs 0.11 dB.
    gt_cam : int | None
        Camera to score against; ``None`` uses ``dst_cam`` (the real photograph of that
        viewpoint), which is how the paper is evaluated.  ``-1`` skips scoring.
    gt_image : (H,W,3) uint8 | str | None
        Score against your own ground-truth image instead of a dataset view.
    config : RunConfig | None
        Any :class:`lfrd.config.RunConfig` field can be set here; the explicit keyword
        arguments above are applied on top of it.
    quiet : bool
        Suppress the pipeline's stdout (it is still captured in ``result.log``).
    strict : bool
        ``True`` (default) raises if any stage exits non-zero.  Set ``False`` to continue when
        a stage only failed its internal DIAGNOSTIC assertions while still writing its output
        -- which is what happens with hand-made depth maps, where checks like "the reference
        foreground mask covers this component" or "stage 5's depth prediction is not a no-op"
        do not apply.  The failed checks are still reported in ``result.stages``.

    Returns
    -------
    FillResult

    Examples
    --------
    >>> from lfrd.api import fill_holes
    >>> res = fill_holes("demo", 5, 4, "f000")                        # paper setting
    >>> res = fill_holes("demo_best", 5, 4, "f000", temporal_frames=100,
    ...                  fill_oofa=True)                              # best measured
    >>> res.save("filled_f000.png")
    >>> print(res.summary())
    """
    # ---- configuration ------------------------------------------------------ #
    if config is None:
        from .config import RunConfig
        config = RunConfig()
    cfg = config
    cfg.src_cam = int(src_cam)
    cfg.dst_cam = int(dst_cam)
    if dataset_root is not None:
        cfg.dataset_root = dataset_root
    cfg.temporal_frames = int(temporal_frames)
    cfg.fill_oofa = bool(fill_oofa)
    cfg.photo_correct = bool(photo_correct)
    cfg.refguide = bool(refguide)
    if gt_cam is not None:
        cfg.gt_cam = None if int(gt_cam) < 0 else int(gt_cam)
    external = ref_image is not None or ref_depth is not None
    if external:
        if ref_image is None or ref_depth is None:
            raise ValueError("pass BOTH ref_image and ref_depth, or neither")
        if not calib_file:
            raise ValueError("ref_image/ref_depth need calib_file (camera parameters)")
        if temporal_frames:
            raise ValueError("temporal_frames needs the dataset sequence; use "
                             "temporal_frames=0 with your own pictures")

    py = python or sys.executable
    frame_run = f"{run}__cam{src_cam}-{dst_cam}-{frame}"

    # ---- stage an external reference view in the layout the stages expect ---- #
    if external:
        img = np.asarray(ref_image) if not isinstance(ref_image, str) else \
            io_utils.imread(ref_image)
        dep = np.asarray(ref_depth) if not isinstance(ref_depth, str) else \
            io_utils.imread(ref_depth, gray=True)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"ref_image must be (H,W,3), got {img.shape}")
        if dep.shape != img.shape[:2]:
            raise ValueError(f"ref_depth {dep.shape} != image {img.shape[:2]}")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if dep.dtype != np.uint8:
            dep = np.clip(np.rint(dep), 0, 255).astype(np.uint8)
        shim = os.path.join(io_utils.OUTPUT_ROOT, "_ext_dataset", run)
        cam_dir = os.path.join(shim, f"cam{src_cam}")
        os.makedirs(cam_dir, exist_ok=True)
        io_utils.imwrite(os.path.join(cam_dir, f"color-cam{src_cam}-{frame}.jpg"), img)
        io_utils.imwrite(os.path.join(cam_dir, f"depth-cam{src_cam}-{frame}.png"), dep)
        basename = os.path.basename(calib_file)
        if not basename.startswith("calibParams-"):
            basename = "calibParams-mine.txt"
        dest = os.path.join(shim, basename)
        if os.path.abspath(calib_file) != os.path.abspath(dest):
            with open(calib_file, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())
        cfg.dataset_root = shim
        cfg.calib_name = basename[len("calibParams-"):-len(".txt")]
        if gt_image is not None:
            g = np.asarray(gt_image) if not isinstance(gt_image, str) else \
                io_utils.imread(gt_image)
            gdir = os.path.join(shim, f"cam{dst_cam}")
            os.makedirs(gdir, exist_ok=True)
            io_utils.imwrite(os.path.join(gdir, f"color-cam{dst_cam}-{frame}.jpg"),
                             np.clip(g, 0, 255).astype(np.uint8))
            cfg.gt_cam = int(dst_cam)

    # ---- drive the stage scripts -------------------------------------------- #
    cfg_path = os.path.join(io_utils.OUTPUT_ROOT, "_cfg_api", f"{run}.json")
    cfg.save(cfg_path)
    log: list = []
    stages: list = []
    args = ["--run", frame_run, "--src_cam", str(src_cam), "--dst_cam", str(dst_cam),
            "--frame", frame, "--config", cfg_path]
    for script in steps:
        sargs = list(args)
        if fill_oofa and script == "step6_render.py":
            sargs.append("--fill_oofa")
        if script == "step5_inpaint.py":
            sargs += ["--no_ablate", "--no_panel"]
        cmd = [py, os.path.join(_ROOT, script)] + sargs
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=_ROOT)
        out = (p.stdout or "") + (p.stderr or "")
        log.append(f"$ {' '.join(cmd[1:])}\n{out}")
        npass = out.count("[CHECK][PASS]")
        nfail = out.count("[CHECK][FAIL]")
        stages.append(dict(script=script, returncode=int(p.returncode), passed=npass,
                           failed=nfail))
        if not quiet:
            print(f"  {script:24s} exit={p.returncode} checks={npass}/{nfail}")
        if p.returncode != 0 and strict:
            tail = "\n".join(out.strip().splitlines()[-15:])
            raise RuntimeError(f"{script} failed (exit {p.returncode}):\n{tail}")
        if p.returncode != 0 and not quiet:
            print(f"    (non-strict: continuing, {nfail} diagnostic check(s) failed)")

    # ---- collect the result ------------------------------------------------- #
    # the stage scripts share one run directory, so copy this frame's artefacts into the
    # per-frame folder first (exactly what run_all.py does), otherwise a later frame would
    # overwrite the stage directories we are about to read.
    import shutil
    run_root = io_utils.run_dir(run, create=True)
    fd = os.path.join(run_root, f"cam{src_cam}-cam{dst_cam}-{frame}")
    for stage in ("calib", "preproc", "warp", "class", "removal", "fill", "final"):
        s = io_utils.run_dir(frame_run, stage, create=False)
        if os.path.isdir(s):
            _copy_small(s, os.path.join(fd, io_utils.STAGE_DIRS[stage]))
    f_final = os.path.join(fd, "60_final", "final.png")
    if not os.path.isfile(f_final):
        failed = [s["script"] for s in stages if s["returncode"] != 0]
        raise RuntimeError(
            f"stage 6 did not produce {f_final}"
            + (f"; failing stages: {failed}" if failed else ""))
    color = io_utils.imread(f_final)
    w = os.path.join(fd, "20_warp")
    m = lambda name: io_utils.imread(os.path.join(w, name), gray=True) > 0
    hole_all = m("hole_all.png")
    disocc = m("hole_disocc.png")
    crack = m("hole_crack.png")
    oofa = m("hole_oofa.png")
    warp = io_utils.imread(os.path.join(w, "warped_color.png"))
    filled_layer = (color != warp).any(axis=2)
    depth = np.full(hole_all.shape, -1.0, np.float32)
    dp = os.path.join(fd, "60_final", "final_depth.npy")
    if os.path.isfile(dp):
        depth = np.load(dp).astype(np.float32)

    metrics = {}
    try:
        import json
        mp = os.path.join(fd, "60_final", "metrics.json")
        if os.path.isfile(mp):
            with open(mp, encoding="utf-8") as fh:
                raw = json.load(fh)
            for key, val in raw.items():
                if isinstance(val, dict):
                    metrics[f"psnr_{key}"] = float(val.get("ours_psnr", float("nan")))
                    metrics[f"ssim_{key}"] = float(val.get("ours_ssim", float("nan")))
                    metrics[f"psnr_warp_{key}"] = float(val.get("warp_psnr", float("nan")))
    except Exception:                                                # noqa: BLE001
        pass

    res = FillResult(color=color, depth=depth, hole=hole_all, disocclusion=disocc,
                     crack=crack, oofa=oofa, filled_from_layer=filled_layer,
                     valid_untouched=~hole_all, metrics=metrics,
                     run_dir=io_utils.run_dir(run, create=False), frame_dir=fd,
                     log=log, stages=stages)
    return res
