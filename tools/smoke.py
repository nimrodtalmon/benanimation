"""Quick check that clips still run after code changes: one frame of each kind of shot per clip,
skipping close-ups that are not rendered yet. python3 tools/smoke.py [clip names]"""
import os
import runpy
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = sys.argv[1:] or ["distorted", "campfire", "rooftop"]
for name in names:
    clip = f"{ROOT}/clips/{name}/clip.py"
    sys.argv = [clip]
    t0 = time.time()
    g = runpy.run_path(clip)
    show, prod = g["show"], g["prod"]
    seen, rng = set(), np.random.default_rng(0)
    for a, b, w, o in show.shots:
        kind = "wide" if isinstance(w, tuple) else str(w).split("_")[0] if str(w).startswith("cu_") else str(w)
        if kind in seen:
            continue
        if str(w).startswith("cu_"):
            c = show._cast[w]()
            if c.cu is None or not os.path.exists(f"{c.cu.d}/meta.json"):
                continue
        seen.add(kind)
        img = prod.frame((a + b) / 2, 0, rng)
        assert img.shape[:2] == (1080, 1920) and np.isfinite(img).all(), (name, w)
    print(f"{name}: ok, {len(seen)} kinds of shots, {time.time() - t0:.0f} s")
