"""Shared CLI plumbing for the step scripts.

Every step script parses the same core arguments and loads/saves through the same
run-directory contract, so that stages can be run independently or chained.
"""
import argparse
import os

from . import io_utils
from .config import RunConfig, parse_frames

DEFAULT_RUN = "ba54_f000"


def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN, help="run name (output/<run>/)")
    ap.add_argument("--dataset_root", default=io_utils.DATASET_ROOT_DEFAULT)
    ap.add_argument("--src_cam", type=int, default=5, help="reference camera")
    ap.add_argument("--dst_cam", type=int, default=4, help="virtual camera")
    ap.add_argument("--frame", default="f000", help="frame name (f000) or index (0)")
    ap.add_argument("--frames", default=None, help="evaluation frame spec: all | 0,5 | 0-9")
    ap.add_argument("--config", default=None, help="JSON config overriding the defaults")
    ap.add_argument("--out_dir", default=None, help="override the output stage dir")
    ap.add_argument("--no_panel", action="store_true", help="skip the visualisation panel")
    return ap


def frame_of(args):
    """'f000' / '0' -> 'f000'."""
    f = str(args.frame)
    if f.startswith("f"):
        return f
    return io_utils.frame_name(int(f))


def make_config(args, **overrides):
    cfg = RunConfig.load(args.config) if getattr(args, "config", None) else RunConfig()
    cfg.run_name = args.run
    cfg.dataset_root = args.dataset_root
    cfg.src_cam = args.src_cam
    cfg.dst_cam = args.dst_cam
    if getattr(args, "frames", None):
        cfg.frames = parse_frames(args.frames)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def stage_dir(args, stage):
    if getattr(args, "out_dir", None):
        os.makedirs(args.out_dir, exist_ok=True)
        return args.out_dir
    return io_utils.run_dir(args.run, stage)


def ensure_selection(args, cfg):
    """Create/refresh output/<run>/selection.json so later stages know the pairing."""
    path = os.path.join(io_utils.run_dir(args.run), "selection.json")
    obj = {"run": args.run, "src_cam": cfg.src_cam, "dst_cam": cfg.dst_cam,
           "frame": frame_of(args), "dataset_root": cfg.dataset_root}
    io_utils.write_json(path, obj)
    return obj


def run_context(args, cfg):
    """-> (cams, ref_view) with the reference colour/depth loaded."""
    from . import calib
    cams = calib.load_calib(cfg.dataset_root, cfg.calib_name)
    ref = io_utils.load_view(cfg.dataset_root, cfg.src_cam, frame_of(args))
    return cams, ref
