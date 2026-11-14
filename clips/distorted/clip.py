"""Clip 5 'distorted': the band Distorted Reality (Ben's characters) plays their own song (an
instrumental, recorded by the band). Distor on drums, the blonde on guitar, the boy from the band
page on bass.

The show follows the song: the bass alone in the dark under one spotlight (the band's logo fades
in over it), the lights and the logo sign on the back wall come on when the drums come in, the
lights drop out on the stop-time breaks (the players hit the break and freeze), the camera moves
more as the song builds, and the logo returns at the end.

  python3 clips/distorted/clip.py segment 13 23      -> a short piece with audio
  python3 clips/distorted/clip.py stills 2 16 33
  python3 clips/distorted/clip.py video               -> out/distorted.mp4
Song: public/songs/dr1 (score from puppet/transcribe.py on Demucs stems).
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.bust_player import BustPlayer  # noqa: E402
from cast.distor_drums import DistorDrummer  # noqa: E402
from production import H, W, HandDrawn, Looks, Production, Stage, TVStatic  # noqa: E402
from scenery import Haze, Sign, ben_logo, title_card  # noqa: E402
from song import Song  # noqa: E402

song = Song("public/songs/dr1/score.json", "public/songs/dr1/song.mp3", None, "Music: Distorted Reality (the band)")
D = song.data
HOLES = [tuple(h) for h in D["holes"]]
DRUMS_IN = min(d["t"] for d in D["drums"] if d["t"] > 10)
LAST_HIT = max(d["t"] for d in D["drums"])
END = LAST_HIT + 0.25  # the end card comes up after the last hit
CRASH = np.array([d["t"] for d in D["drums"] if d["piece"] == 49])
STOPS = [h for h in HOLES if 60 < h[0] < 70]  # the stop-time section: flashes when the band comes back

HEAD = 0.6
CAST = {
    "drums": lambda: DistorDrummer(song, head=HEAD, face=True, groove=1.0),
    "drums_close": lambda: DistorDrummer(song, scale=0.85, top=-60, head=HEAD, face=True, groove=1.0),
    "guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.40, top=30, head=HEAD, seed=3),
    "guitar_close": lambda: BustPlayer(song, "blonde", "guitar", scale=0.62, top=-60, head=HEAD, seed=3),
    "bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.55, top=40, head=HEAD, seed=5),
    "bass_close": lambda: BustPlayer(song, "bassboy", "bass", scale=0.80, top=-90, head=HEAD, seed=5),
    # the wide shot: drums at the back, guitar and bass in front, bodies running off the bottom
    "w_drums": lambda: DistorDrummer(song, scale=0.36, top=150, head=HEAD, face=True, groove=1.0),
    "w_guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.25, cx=W * 0.24, top=400, head=HEAD, seed=3),
    "w_bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.34, cx=W * 0.77, top=330, head=HEAD, seed=5),
    # the two front players side by side
    "d_guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.33, cx=W * 0.30, top=170, head=HEAD, seed=3),
    "d_bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.45, cx=W * 0.72, top=110, head=HEAD, seed=5),
}
WIDE = ("w_drums", "w_guitar", "w_bass")
DUO = ("d_guitar", "d_bass")


def pan(x0, x1, a, b, y=H * 0.42):
    return lambda t: (x0 + (x1 - x0) * min(1.0, max(0.0, (t - a) / (b - a))), y)


def shot_list():
    """Cuts on bar lines. Intro: the bass alone, two bars per shot, slow moves. From the drums on:
    a cycle of angles, one bar each, each with its own small move (push in, pull out, pan)."""
    bars = D["beats"][::4] + [song.duration + 1]
    intro = [("bass", dict(zoom=(1.0, 1.06))), (WIDE, "pan"), ("bass_close", dict(zoom=(1.0, 1.08))),
             ("bass", dict(zoom=(1.08, 1.0))), (WIDE, "pan"), ("bass_close", dict(zoom=(1.04, 1.1))), ("bass", dict(zoom=(1.0, 1.05)))]
    cycle = [("drums", dict(zoom=(1.12, 1.0))), (WIDE, "pan"), ("guitar", dict(zoom=(1.0, 1.07))), ("bass", dict(zoom=(1.07, 1.0))),
             ("drums_close", dict(zoom=(1.0, 1.06))), (DUO, "pan"), ("guitar_close", dict(zoom=(1.06, 1.0))), (WIDE, "pan"),
             ("bass_close", dict(zoom=(1.0, 1.07))), ("drums", dict(zoom=(1.0, 1.05))), (DUO, "pan"), ("guitar", dict(zoom=(1.07, 1.0)))]
    shots, i, k, kc, flip = [], 0, 0, 0, 1
    t0 = 0.0
    while i < len(bars) - 1:
        if bars[i] < DRUMS_IN - 0.3:  # intro: two bars per shot
            j = min(i + 2, len(bars) - 1)
            while bars[j] > DRUMS_IN - 0.3 and j > i + 1:
                j -= 1
            who, opts = intro[k % len(intro)]
            k += 1
        else:
            j = i + 1
            who, opts = cycle[kc % len(cycle)]
            kc += 1
        t1 = bars[j]
        if opts == "pan":
            flip = -flip
            x0, x1 = (W * 0.40, W * 0.60) if flip > 0 else (W * 0.60, W * 0.40)
            opts = dict(focus=pan(x0, x1, t0, t1), zoom=(1.06, 1.06))
        shots.append((t0, t1, who, opts))
        t0, i = t1, j
    return shots


SHOTS = shot_list()


def spot_x(shot):
    """Where the bassist stands in this shot (the intro spotlight follows him)."""
    if shot == WIDE:
        return W * 0.77
    if shot == DUO:
        return W * 0.72
    return W / 2 if isinstance(shot, str) and shot.startswith("bass") else None


def hole_at(t):
    for a, b in HOLES:
        if a <= t < b:
            return a, b
    return None


def cue(t, shot):
    """Lights: the intro under one spot; full color from the drums on; dark on the breaks; out at the end."""
    if t < DRUMS_IN:
        x = spot_x(shot)
        k = 0.35 if hole_at(t) else 1.0
        return dict(beams=0.12, house=0.55 * k, spot=(x, 1.3 * k) if x is not None else None)
    if t >= END:
        return dict(beams=0.0, house=max(0.0, 1 - (t - END) / 0.6))
    if hole_at(t):
        return dict(beams=0.0, house=0.3)
    return dict(beams=1.0, house=1.0)


logo = ben_logo()
SIGN = Sign(logo, width=820, y=70)
HAZE = Haze()
RED = (0.30, 0.25, 1.0)


def backdrop(frame, t, shot):
    """The logo sign on the back wall: off in the intro, flashes on with the drums, takes the color of
    the bar's lights, glows red on the breaks."""
    if t < DRUMS_IN - 0.05 or t >= END + 0.4:
        return
    on = min(1.0, (t - DRUMS_IN + 0.05) / 0.08)
    bar = int(t / song.bar) % 4
    col = RED if hole_at(t) else tuple(0.5 + 0.5 * np.array(Stage.COLORS[bar]))
    SIGN.draw(frame, t, col, 0.8 * on)


def fore(frame, t, shot):
    """Haze over the floor (thicker in the intro), and white flashes: the drums coming in and the
    band coming back after each stop."""
    HAZE.draw(frame, t, 1.4 if t < DRUMS_IN else 0.8)
    flash = 0.0
    for t_on in [DRUMS_IN] + [b for a, b in STOPS]:
        u = t - t_on
        if 0 <= u < 0.4:
            flash = max(flash, math.exp(-u / 0.09))
    if flash > 0:
        frame += 0.35 * flash


def post(img, t):
    """The logo over the opening, and as the end card with the credits."""
    if t < 4.2:
        k = min(1.0, max(0.0, (t - 0.4) / 0.9)) * min(1.0, max(0.0, (4.0 - t) / 0.8))
        title_card(img, logo, 0.85 * k)
    elif t >= END:
        k = min(1.0, (t - END) / 0.5)
        title_card(img, logo, k, sub=["music: Distorted Reality", "drawings: Ben"])
    return img


prod = Production(song, CAST, SHOTS, name="distorted", script=os.path.abspath(__file__),
                  look=Looks(HandDrawn(), TVStatic(0.05, vhs=0.25)), stage="stage", crowd_wide=False, cue=cue,
                  drift=lambda a: 0.05 if a < 44 else (0.065 if a < 68 else 0.085),  # more camera as the song builds
                  parallax=0.55, ghost=0.35, punch=(CRASH, 0.04), roll=0.6, backdrop=backdrop, fore=fore, post=post)

if __name__ == "__main__":
    prod.main(sys.argv)
