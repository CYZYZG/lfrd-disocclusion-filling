"""lfrd -- Local Foreground Removal Disocclusion filling (paper reproduction).

Paper: H. Liang et al., "Local Foreground Removal Disocclusion Filling Method for View
Synthesis", IEEE Access 8:201286-201299, 2020.

Stage modules
-------------
preprocess.py     step 1  morphology-based depth preprocessing (ghost removal)
warp.py           step 2  forward 3D warping (+ crack handling, backward index)
disocclusion.py   step 2/3 hole typing + disocclusion edge FG/BG classification
fill.py           step 4/5.1 local foreground removal + removed-region depth prediction
inpaint.py        step 5.2 modified Criminisi (priority + depth-limited patch search)
render.py         step 6  disocclusion filling, postprocessing, two-view fusion
pipeline.py       end-to-end driver used by run_all.py
"""
from .config import RunConfig, parse_frames

__all__ = ["RunConfig", "parse_frames"]
