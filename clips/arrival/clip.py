"""Clip 3, 'Arrival': an empty stage, a drum kit and a mic stand. Distor walks in from the left,
Pepper from the right (behind the kit); a cut hides her sitting down; she counts in on the
hi-hat and the song starts. Toolkit features: Song.shifted (a scene before the music, count-in,
footsteps), shared walking legs (puppet/legs.py), fluid motion, the HandDrawn finish.

  python3 clips/arrival/clip.py stills 1 2.5 3.5 5 8 20
  python3 clips/arrival/clip.py segment 0 10      -> out/arrival-0-10.mp4 (the walk-in)
  python3 clips/arrival/clip.py video             -> out/arrival.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.distor import Singer  # noqa: E402
from cast.pepper import Drummer  # noqa: E402
from cast.pepper_walk import PepperWalk  # noqa: E402
from legs import contacts  # noqa: E402
from motion import Path  # noqa: E402
from production import H, W, HandDrawn, Production  # noqa: E402
from song import Song  # noqa: E402

PRE = 6.0  # seconds of scene before the music (3 bars, so the beat grid stays)
FLOOR = 1000  # screen y of the soles in the wide shot (front of the stage)
D_SCALE, D_STOP = 0.245, 620
P_SCALE, P_STOP, KIT_X = 0.40, 1330, 1330
d_walk = Path([(0, -260), (0.2, -260), (3.6, D_STOP)], ramp=0.6)
p_walk = Path([(0, 2250), (0.6, 2250), (3.7, P_STOP)], ramp=0.5)
STAND = dict(x=D_STOP - 235, grab=4.5)
D_TOP = FLOOR - 3890 * D_SCALE
KIT_TOP = FLOOR - 1760 * P_SCALE


def make_song():
    base = Song.default()
    steps = [(t, "step", 0.35) for t, _ in contacts(Singer(base, scale=D_SCALE, top=D_TOP, walk=d_walk, fluid=True).walker, 0, PRE)]
    steps += [(t, "step", 0.25) for t, _ in PepperWalk(base, scale=P_SCALE, path=p_walk, floor_y=FLOOR - 30).legs.contacts(0, PRE)]
    return base.shifted(PRE, name="arrival", count_in=(4.0, 4), sfx=steps)


song = make_song()

CAST = {
    "dis_wide": lambda: Singer(song, scale=D_SCALE, top=D_TOP, walk=d_walk, stand=STAND, fluid=True, head=0.6, mouth_twos=True),
    "pep_walk": lambda: PepperWalk(song, scale=P_SCALE, path=p_walk, floor_y=FLOOR - 30),
    "kit": lambda: Drummer(song, scale=P_SCALE, cx=KIT_X, top=KIT_TOP, empty=True),
    "drums_wide": lambda: Drummer(song, scale=P_SCALE, cx=KIT_X, top=KIT_TOP),
    "drummer": lambda: Drummer(song),
    "singer": lambda: Singer(song, fluid=True, head=0.6, mouth_twos=True),
    "singer_close": lambda: Singer(song, scale=0.62, top=-10, fluid=True, head=0.6, mouth_twos=True),
}
WALK_IN = ("pep_walk", "kit", "dis_wide")
BAND = ("drums_wide", "dis_wide")
S = PRE  # song time 0
SHOTS = [
    (0, 3.9, WALK_IN, dict(zoom=(1.0, 1.04))),
    (3.9, S, "drummer", dict(zoom=(1.08, 1.12))),  # she counts in on the hi-hat
    (S, S + 4, BAND, dict(zoom=(1.04, 1.0))),
    (S + 4, S + 8, "drummer"), (S + 8, S + 9.6, BAND),
    (S + 9.6, S + 14, "singer"), (S + 14, S + 16, "singer_close"), (S + 16, S + 18, "drummer"), (S + 18, S + 20, BAND),
    (S + 20, S + 22, "singer"), (S + 22, S + 24, BAND, dict(zoom=(1.0, 1.06))), (S + 24, S + 26, "drummer"),
    (S + 26, S + 30, "singer_close"), (S + 30, S + 32, BAND), (S + 32, S + 34, "drummer"), (S + 34, S + 38, "singer"),
    (S + 38, S + 40, "singer_close"), (S + 40, S + 44, BAND, dict(zoom=(1.0, 1.08))),
]


def ramp(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def cue(t, shot):
    beams = ramp(t, S - 0.2, S + 0.6) * (1 - ramp(t, S + 42.0, S + 43.5))
    spot_x = d_walk.x(t) - 40 if shot in (WALK_IN, BAND) else W / 2
    return dict(beams=beams, spot=(spot_x, 0.75 * ramp(t, 0.0, 1.0)), house=0.7 + 0.3 * beams)


def pep():  # for tools/jerk.py
    return CAST["pep_walk"]()


def dis():
    return CAST["dis_wide"]()


# the camera as in Clint Eastwood: a slow constant drift in every shot, the back wall moving less
# than the band (parallax), frame blending on fast motion (sticks)
prod = Production(song, CAST, SHOTS, name="arrival", script=os.path.abspath(__file__), cue=cue,
                  look=HandDrawn(), crowd_wide=False, drift=0.05, parallax=0.55, ghost=0.35)

if __name__ == "__main__":
    prod.main(sys.argv)
