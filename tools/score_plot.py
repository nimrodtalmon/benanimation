"""Check a transcription by eye: band levels of the drum and bass stems with the detected hits.

  python3 tools/score_plot.py <score.json> <stems dir> <t0> <t1> <out.png>
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "puppet"))
import transcribe as T  # noqa: E402

score, stems, t0, t1, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
s = json.load(open(score))
y = T.load(f"{stems}/drums.wav")
rows = [("kick (low)", T.band(y, None, 110), [36], None), ("snare (mid)", T.band(y, 160, 900), [38], None),
        ("hat / crash (high)", T.band(y, 6000, 10500), [42, 46, 49], None), ("bass", T.load(f"{stems}/bass.wav"), None, "bass"),
        ("guitar (other)", T.load(f"{stems}/other.wav"), None, "guitar")]
fig, ax = plt.subplots(len(rows), 1, figsize=(16, 2.2 * len(rows)), sharex=True)
for a, (name, yy, pcs, key) in zip(ax, rows):
    e = T.env(yy)
    tt = np.arange(len(e)) * T.HOP / T.SR
    m = (tt > t0) & (tt < t1)
    a.plot(tt[m], e[m], lw=1)
    a.set_ylabel(name)
    ev = [d for d in s["drums"] if d["piece"] in pcs] if pcs else s[key]
    for d in ev:
        if t0 < d["t"] < t1:
            a.axvline(d["t"], color="g" if d.get("piece") == 49 else "r", lw=1)
    for b in s.get("beats", []):
        if t0 < b < t1:
            a.axvline(b, color="k", ls=":", lw=0.6)
plt.tight_layout()
plt.savefig(out, dpi=70)
