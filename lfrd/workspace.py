"""Where the data and the sibling project live -- resolved from the environment, not hard-coded.

The reproduction needs three things that are NOT redistributed with the code:

  * the MSR 3D Video Ballet dataset (research use only)
  * the official calibration file that ships inside that dataset
  * (optional) the sibling reproduction this repo is compared against in the reports

Resolve them in this order, so a clone works without editing any file:

  1. an explicit path passed by the caller / command line
  2. an environment variable
  3. ``<project>/data/...`` next to the code

Environment variables
---------------------
``BALLET_DATA_ROOT``   dataset root containing ``cam0..cam7`` and ``calibParams-ballet.txt``
``SIBLING_ROOT``       the sibling DIBR reproduction (only needed by the head-to-head tools)
``LFRD_PYTHON``        interpreter used when a tool shells out (defaults to ``sys.executable``)

    from lfrd import workspace
    root = workspace.dataset_root()          # raises a helpful error if unset
    sib  = workspace.sibling_root()          # returns None when not configured
"""
from __future__ import annotations

import os
import sys

__all__ = ["project_root", "output_root", "data_root", "dataset_root", "default_dataset_root",
           "sibling_root", "python_exe", "env_report", "DATASET_ENV", "SIBLING_ENV",
           "PYTHON_ENV"]

DATASET_ENV = "BALLET_DATA_ROOT"
SIBLING_ENV = "SIBLING_ROOT"
PYTHON_ENV = "LFRD_PYTHON"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_root() -> str:
    """The checkout this module lives in."""
    return _PROJECT_ROOT


def output_root() -> str:
    """Where runs, panels and inventories are written (``<project>/output``)."""
    return os.path.join(_PROJECT_ROOT, "output")


def data_root() -> str:
    """Local data directory next to the code (``<project>/data``)."""
    return os.path.join(_PROJECT_ROOT, "data")


def _first_existing(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


def default_dataset_root() -> str:
    """Best guess for the dataset root, resolved WITHOUT raising (import-safe).

    Used to seed the library's module-level default so that an existing local checkout keeps
    working without setting anything, while a fresh clone still gets a clear error later.
    The dataset folder is recognised by the presence of ``calibParams-ballet.txt``.
    """
    cands = [
        os.environ.get(DATASET_ENV),
        os.path.join(data_root(), "MSR3DVideo-Ballet"),
        data_root(),
    ]
    # conventional sibling location: a "3DVideos-distrib" folder next to the project
    parent = os.path.dirname(_PROJECT_ROOT)
    grandparent = os.path.dirname(parent)
    for base in (parent, grandparent):
        if base:
            cands.append(os.path.join(base, "3DVideos-distrib", "MSR3DVideo-Ballet"))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "calibParams-ballet.txt")):
            return c
    # nothing found: return the conventional local path so the error message is instructive
    return os.path.join(data_root(), "MSR3DVideo-Ballet")


def dataset_root(required: bool = False):
    """MSR Ballet root.

    Order: ``BALLET_DATA_ROOT`` -> ``<project>/data/MSR3DVideo-Ballet`` ->
    ``<project>/data`` -> a ``3DVideos-distrib`` folder next to the project.  With
    ``required=True`` this raises instead of returning None, and the message says what to set.
    """
    cands = [os.environ.get(DATASET_ENV),
             os.path.join(data_root(), "MSR3DVideo-Ballet"),
             data_root()]
    parent = os.path.dirname(_PROJECT_ROOT)
    for base in (parent, os.path.dirname(parent)):
        if base:
            cands.append(os.path.join(base, "3DVideos-distrib", "MSR3DVideo-Ballet"))
    found = _first_existing(*cands)
    if found is None and required:
        raise FileNotFoundError(
            f"MSR Ballet dataset not found.  Set the environment variable {DATASET_ENV} to its "
            f"root (the folder that contains cam0..cam7 and calibParams-ballet.txt), or put it "
            f"at {os.path.join(data_root(), 'MSR3DVideo-Ballet')}.  The dataset is not "
            f"redistributed with this repository (research use only).")
    return found


def sibling_root(required: bool = False):
    """The sibling DIBR reproduction used by the head-to-head tools; None when not configured."""
    found = _first_existing(os.environ.get(SIBLING_ENV),
                            os.path.join(data_root(), "sibling"))
    if found is None and required:
        raise FileNotFoundError(
            f"sibling project not found.  Set {SIBLING_ENV} to its root, or put it at "
            f"{os.path.join(data_root(), 'sibling')}.  Only the head-to-head comparison tools "
            f"need it.")
    return found


def python_exe() -> str:
    """Interpreter for tools that shell out to the stage scripts."""
    return os.environ.get(PYTHON_ENV) or sys.executable


def env_report() -> str:
    """Human-readable resolution status; handy as the first line of a tool's output."""
    dr = dataset_root()
    sr = sibling_root()
    lines = [f"project root : {project_root()}",
             f"{DATASET_ENV:16s}: {dr or '(unset -- dataset-dependent tools will refuse to run)'}",
             f"{SIBLING_ENV:16s}: {sr or '(unset -- head-to-head tools will refuse to run)'}",
             f"{PYTHON_ENV:16s}: {python_exe()}"]
    return "\n".join(lines)
