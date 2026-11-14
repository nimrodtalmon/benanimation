"""Smoothness check for a character: draws it frame by frame (no camera) and reports, per traced
key point, the largest per-frame acceleration (px/frame^2) and when it happens. Spikes = pops.

  python3 tools/jerk.py <module:factory> t0 t1     e.g. clips/motiontest/clip.py:singer
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/puppet")
from production import FPS, H, W  # noqa: E402


def trace(make, t0, t1):
    ch = make()
    frame = np.zeros((H, W, 3), np.float32)
    rows = []
    for f in range(int(t0 * FPS), int(t1 * FPS)):
        frame[:] = 0
        ch.draw(frame, f / FPS)
        rows.append({k: np.asarray(v, float)[:2] for k, v in ch.trace.items()})
    return rows


def report(rows, t0, top=3):
    keys = sorted(set.intersection(*[set(r) for r in rows]))
    out = {}
    for k in keys:
        P = np.array([r[k] for r in rows])
        acc = np.hypot(*(P[2:] - 2 * P[1:-1] + P[:-2]).T)
        i = np.argsort(acc)[::-1][:top]
        out[k] = (float(np.percentile(acc, 95)), [(round(t0 + (j + 1) / FPS, 2), round(float(acc[j]), 1)) for j in i])
    return out


if __name__ == "__main__":
    path, name = sys.argv[1].split(":")
    spec = importlib.util.spec_from_file_location("clipmod", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t0, t1 = float(sys.argv[2]), float(sys.argv[3])
    for k, (p95, worst) in report(trace(getattr(m, name), t0, t1), t0).items():
        print(f"{k:12s} p95={p95:5.1f}  worst={worst}")
