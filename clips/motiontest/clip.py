"""Motion test: Distor walks a few steps into the spot, stops, lifts the mic and sings one line.
The same moment in several options, to choose how characters should move and look.

  python3 clips/motiontest/clip.py A|B|C|D      -> out/motiontest-<opt>.mp4
  python3 clips/motiontest/clip.py grid         -> out/motiontest-grid.mp4 (all four, 2x2, labeled)

  A  current toolkit (baseline)
  B  fluid motion: weight shift, overlapping arms, hand drag, lean and settle
  C  B + hand-drawn finish (line boil, paper grain, color grade)
  D  C with the characters on twos (15 drawings a second, smooth camera)
  E  B + subtle finish (the default HandDrawn: a faint boil)
  pair                                          -> out/motiontest-BE.mp4 (B and E side by side)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.distor import Singer  # noqa: E402
from motion import Path  # noqa: E402
from production import H, W, HandDrawn, Production  # noqa: E402
from song import Song  # noqa: E402

song = Song.default()
T0, T1 = 6.6, 14.2  # walk 6.8-9.4, stop, the line starts ~10.2
walk = Path([(0, 380), (6.8, 380), (9.4, 1010)], ramp=0.5)
OPTS = {
    "A": dict(fluid=False, look=None, twos=False),
    "B": dict(fluid=True, look=None, twos=False),
    "C": dict(fluid=True, look="hd", twos=False),
    "D": dict(fluid=True, look="hd", twos=True),
    "E": dict(fluid=True, look="subtle", twos=False),
}
LABEL = {"A": "A  now", "B": "B  fluid motion", "C": "C  fluid + hand-drawn finish", "D": "D  C on twos", "E": "E  fluid + subtle finish"}


def cue(t, shot):
    return dict(beams=0.0, spot=(max(walk.x(t) - 40, 760), 0.9), house=0.75)


def follow(t):
    return (W / 2 + max(-90, min(90, 0.12 * (walk.x(t) - W / 2))), H * 0.40)


def production(opt):
    o = OPTS[opt]
    cast = {"full": lambda: Singer(song, scale=0.25, top=50, walk=walk, fluid=o["fluid"])}
    shots = [(0, song.duration, "full", dict(zoom=(1.12, 1.16), focus=follow))]
    look = {None: None, "hd": HandDrawn(boil=1.3, hold=3), "subtle": HandDrawn()}[o["look"]]
    return Production(song, cast, shots, name=f"motiontest-{opt}", script=os.path.abspath(__file__), cue=cue, look=look, twos=o["twos"])


def grid():
    outs = [f"{ROOT}/out/motiontest-{k}.mp4" for k in "ABCD"]
    labels = [LABEL[k].replace(":", "") for k in "ABCD"]
    flt = ";".join(f"[{i}:v]scale=960:540,drawtext=text='{labels[i]}':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5[v{i}]" for i in range(4))
    flt += ";[v0][v1]hstack[top];[v2][v3]hstack[bot];[top][bot]vstack[out]"
    cmd = ["ffmpeg", "-loglevel", "error", "-y"] + sum([["-i", o] for o in outs], []) + [
        "-filter_complex", flt, "-map", "[out]", "-map", "0:a", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", f"{ROOT}/out/motiontest-grid.mp4"]
    subprocess.run(cmd, check=True)


def pair(a="B", b="E"):
    outs = [f"{ROOT}/out/motiontest-{k}.mp4" for k in (a, b)]
    flt = ";".join(f"[{i}:v]scale=960:540,drawtext=text='{LABEL[k]}':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5[v{i}]" for i, k in enumerate((a, b)))
    flt += ";[v0][v1]hstack[out]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", outs[0], "-i", outs[1], "-filter_complex", flt, "-map", "[out]", "-map", "0:a",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", f"{ROOT}/out/motiontest-{a}{b}.mp4"], check=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "grid":
        grid()
    elif what == "pair":
        pair(*sys.argv[2:4])
    elif what == "stills":
        production(sys.argv[2]).stills([float(a) for a in sys.argv[3:]], f"{ROOT}/out/motiontest-{sys.argv[2]}-still")
    else:
        print(production(what).render_range(T0, T1, f"{ROOT}/out/motiontest-{what}.mp4"))
