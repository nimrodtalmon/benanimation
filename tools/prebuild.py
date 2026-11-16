"""Build everything a clip will need from Blender before rendering the video (close-ups, and through
them the instruments/kit; venues and kits build on first use too): python3 tools/prebuild.py clips/<name>/clip.py"""
import os
import runpy
import sys

clip = os.path.abspath(sys.argv[1])
sys.argv = [clip]
g = runpy.run_path(clip)
show = g["show"]
for a, b, w, o in show.shots:
    if str(w).startswith("cu_"):
        c = show._cast[w]()
        if c.cu is not None:
            c.cu.ensure()
            print("ready", w, f"{a:.1f}-{b:.1f}", flush=True)
print("all built")
