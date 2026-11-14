"""Clip 2, 'Soundcheck': an empty stage, a mic on a stand in a single spotlight. Distor walks in,
takes the mic, sings; on the chorus the stage lights come up; at the end only the spot is left.
Same song as clip 1. Uses only toolkit features (walk, stand + grab, light cues, camera pans).

  python3 clips/soundcheck/clip.py stills 1 5 9 12 21 43
  python3 clips/soundcheck/clip.py video     -> out/soundcheck.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.distor import Singer  # noqa: E402
from motion import Path  # noqa: E402
from production import H, W, Production  # noqa: E402
from song import Song  # noqa: E402

song = Song.default()

STOP = 1010  # where he stops (screen x of his center, full shot)
walk_in = Path([(0, -320), (2.2, -320), (7.6, STOP)])
STAND = dict(x=STOP - 235, grab=8.8)  # the stand on his right (viewer's left); he takes the mic at 8.8 s
CHORUS = 20.0

CAST = {
    "full": lambda: Singer(song, scale=0.245, top=40, walk=walk_in, stand=STAND, pump_after=CHORUS),
    "medium": lambda: Singer(song, pump_after=CHORUS),
    "close": lambda: Singer(song, scale=0.62, top=-10, pump_after=CHORUS),
}


def follow(t):  # a gentle pan with him while he walks (zoomed in enough to have room)
    x = walk_in.x(t)
    return (W / 2 + max(-110, min(110, 0.1 * (x - W / 2))), H * 0.42)


SHOTS = [
    (0, 9.6, "full", dict(zoom=(1.15, 1.12), focus=follow)),  # he walks in and takes the mic
    (9.6, 14, "medium"), (14, 16, "close"), (16, 18, "full", dict(zoom=(1.0, 1.04))), (18, 20, "medium"),
    (20, 22, "full", dict(zoom=(1.06, 1.0))),  # lights up
    (22, 26, "medium"), (26, 28, "close"), (28, 30, "full"), (30, 34, "medium"), (34, 38, "close"),
    (38, 42, "medium"), (42, 44, "full", dict(zoom=(1.0, 1.08))),  # lights down
]


def ramp(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def cue(t, shot):
    beams = ramp(t, CHORUS - 0.2, CHORUS + 0.6) * (1 - ramp(t, 42.0, 43.5))
    # the spot waits on the empty stand; he walks into it and it stays with him
    spot_x = max(walk_in.x(t) - 60, STAND["x"] + 120) if shot == "full" else W / 2
    return dict(beams=beams, spot=(spot_x, 0.9 * ramp(t, 0.2, 1.8)), house=0.55 + 0.45 * max(ramp(t, 0.0, 2.0) * 0.5, beams))


prod = Production(song, CAST, SHOTS, name="soundcheck", script=os.path.abspath(__file__), cue=cue)

if __name__ == "__main__":
    prod.main(sys.argv)
