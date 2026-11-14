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

HEAD = 0.45  # Clint Eastwood: heads nearly still (the drummer's too)
CAST = {
    "drums": lambda: DistorDrummer(song, head=HEAD, face=True, groove=0.0),
    "drums_close": lambda: DistorDrummer(song, scale=0.85, top=-60, head=HEAD, face=True, groove=0.0),
    "guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.40, top=30, head=HEAD, seed=3, style="ce"),
    "guitar_close": lambda: BustPlayer(song, "blonde", "guitar", scale=0.62, top=-60, head=HEAD, seed=3, style="ce"),
    "bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.55, top=40, head=HEAD, seed=5, style="ce"),
    "bass_close": lambda: BustPlayer(song, "bassboy", "bass", scale=0.80, top=-90, head=HEAD, seed=5, style="ce"),
    # the wide shot: drums at the back, guitar and bass in front, bodies running off the bottom
    "w_drums": lambda: DistorDrummer(song, scale=0.36, top=150, head=HEAD, face=True, groove=0.0),
    "w_guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.25, cx=W * 0.24, top=400, head=HEAD, seed=3, style="ce"),
    "w_bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.34, cx=W * 0.77, top=330, head=HEAD, seed=5, style="ce"),
    # the two front players side by side
    "d_guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.33, cx=W * 0.30, top=170, head=HEAD, seed=3, style="ce"),
    "d_bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.45, cx=W * 0.72, top=110, head=HEAD, seed=5, style="ce"),
}
WIDE = ("w_drums", "w_guitar", "w_bass")
DUO = ("d_guitar", "d_bass")


def hole_in(a, b):
    return [h for h in HOLES if a - 0.05 <= h[0] < b]


def freeze_at(t0):
    return lambda t: t0


def rock(a, spb):
    """Clint Eastwood's tape trick over one bar: beats 1-2 as played, beat 3 runs backward over beat 2,
    beat 4 plays beat 2 forward again (the audio keeps going)."""
    def f(t):
        u = t - a
        if u < 2 * spb:
            return t
        if u < 3 * spb:
            return a + 2 * spb - (u - 2 * spb)
        return a + spb + (u - 3 * spb)
    return f


def shot_list():
    """Cut like Clint Eastwood (measured from the video): a calm opening with long shots (two bars,
    ~3.8 s) and a still camera; one bar per shot once the band plays; freeze frames on the stop-time
    breaks; twice a bar played forward/backward; flash cuts (a half bar, then two quarter bars) in
    the last part, where the camera also moves most."""
    spb = song.spb
    bars = D["beats"][::4] + [song.duration + 1]
    intro = ["bass", WIDE, "bass_close", "bass", WIDE, "bass_close", "bass"]
    cycle = ["drums", WIDE, "guitar", "bass", "drums_close", DUO, "guitar_close", WIDE, "bass_close", "drums", DUO, "guitar"]
    flash = ["guitar_close", "drums_close", "bass_close", "drums", "guitar", "bass"]
    shots, i, k, kc, kf, nrock = [], 0, 0, 0, 0, 0
    t0 = 0.0
    while i < len(bars) - 1:
        if bars[i] < DRUMS_IN - 0.3:  # intro: two bars per shot, slow pushes
            j = min(i + 2, len(bars) - 1)
            while bars[j] > DRUMS_IN - 0.3 and j > i + 1:
                j -= 1
            shots.append((t0, bars[j], intro[k % len(intro)], dict(zoom=(1.0, 1.04) if k % 2 == 0 else (1.04, 1.0))))
            k += 1
            t0, i = bars[j], j
            continue
        t1 = bars[i + 1]
        who = cycle[kc % len(cycle)]
        kc += 1
        hs = hole_in(t0, t1)
        if hs and t0 >= DRUMS_IN + 1:  # a break inside this bar: play up to it, then freeze on its first frame
            a_, b_ = hs[0]
            shots.append((t0, a_, who, {}))
            shots.append((a_, min(t1, b_ + 0.02), who, dict(tmap=freeze_at(a_ + 0.03))))
            if b_ + 0.02 < t1:
                shots.append((b_ + 0.02, t1, cycle[kc % len(cycle)], {}))
                kc += 1
        elif 68 <= t0 < END - 2:  # flash cuts
            q = (t1 - t0) / 4
            for (u0, u1) in ((0, 2), (2, 3), (3, 4)):
                shots.append((t0 + u0 * q, t0 + u1 * q, flash[kf % len(flash)], {}))
                kf += 1
        elif 46 <= t0 < 60 and who in (WIDE, DUO) and nrock < 2:  # the tape trick, twice
            shots.append((t0, t1, who, dict(tmap=rock(t0, spb))))
            nrock += 1
        else:
            shots.append((t0, t1, who, {}))
        t0, i = t1, i + 1
    return [sh for sh in shots if sh[1] - sh[0] > 0.02]


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
                  drift=lambda a: 0.015 if a < DRUMS_IN else (0.05 if a < 44 else (0.065 if a < 68 else 0.09)),  # camera energy climbs
                  parallax=0.55, ghost=0.35, backdrop=backdrop, fore=fore, post=post)

if __name__ == "__main__":
    prod.main(sys.argv)
