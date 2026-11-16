"""Time per video frame of a clip (body shots and the first close-up), and how many close-ups are
already rendered: python3 tools/profile_clip.py <clip name> [t0 t1]"""
import os
import runpy
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1]
t0, t1 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (18.0, 30.0)
clip = f"{ROOT}/clips/{name}/clip.py"
sys.argv = [clip]
g = runpy.run_path(clip)
show, prod = g["show"], g["prod"]
cus = [(a, b, w) for a, b, w, o in show.shots if str(w).startswith("cu_")]
ready = [w for a, b, w in cus if show._cast[w]().cu is not None and os.path.exists(f"{show._cast[w]().cu.d}/meta.json")]
print(f"close-ups: {len(cus)}, rendered: {len(ready)}")
rng = np.random.default_rng(0)
ts = np.arange(t0, t1, 0.4)
prod.frame(t0, 0, rng)
t = time.time()
for x in ts:
    prod.frame(x, 0, rng)
print(f"shots {t0:g}-{t1:g} s: {(time.time() - t) / len(ts):.2f} s/frame")
for a, b, w in cus:
    if w in ready:
        xs = np.arange(a, b, 0.1)
        t = time.time()
        for x in xs:
            prod.frame(x, 0, rng)
        print(f"close-up {w}: {(time.time() - t) / len(xs):.2f} s/frame")
        break
