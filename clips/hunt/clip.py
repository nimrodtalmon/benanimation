"""Clip 'hunt': a story, not a band show. Distor is a hunter who comes into the woods where the band
(the blonde on guitar, the boy on bass) plays as part of nature; he shoots, the woods catch fire, the
sky turns red; nature strikes back with a storm and lightning hits him; in the quiet he rises from
the ash with glowing eyes; in the finale he stands over the burning forest like a god.

Acts follow the song (public/songs/hunt): the eye (0-5 s), nature (5-40), the hunt (40-109), the
storm (109-144), ash (144-177), the god (177-225), the end. The venue (puppet/wilds.py) does the
light, the sky and the weather from the moods/weather scores below; the shot list is written here.

  python3 clips/hunt/clip.py stills 2 20 42 120 150 190
  python3 clips/hunt/clip.py video                -> out/hunt.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
import numpy as np  # noqa: E402

from motion import Path  # noqa: E402
from production import H, W  # noqa: E402
from scenery import ben_logo, title_card  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402
from venues import Walking  # noqa: E402
from wilds import Wilds  # noqa: E402

SONG = Song.load("hunt")
BAND = [Member("guitar", "blonde", side="blonde_side", sings=True), Member("bass", "bassboy", seed=1)]

# ---- the story's moments (seconds, from the song)
STOP = (4.32, 5.17)  # the first stop: he stares
HUNT, STORM, STRIKE, ASH, GOD, END = 40.18, 109.24, 144.196, 144.58, 176.95, 224.6
GUN = [40.18, 44.78, 54.71, 66.06, 75.0, 84.29, 92.73, 100.0, 107.15, 115.11, 119.79, 126.35, 133.58, 139.2]
crash = [e["t"] for e in SONG.data["drums"] if e["piece"] == 49]
BOLTS = [(t, None) for t in crash if STORM < t < STRIKE - 1][::2] + [(STRIKE, W / 2), (170.68, W * 0.3), (173.19, W * 0.7)] + \
        [(t, None) for t in crash if GOD < t < END][::3]
BIRDS = [(40.18, (W * 0.5, H * 0.6), 1), (54.71, (W * 0.3, H * 0.62), -1), (75.0, (W * 0.65, H * 0.6), 1), (100.0, (W * 0.4, H * 0.62), -1)]

MOODS = [(0.0, "night"), (5.17, "dawn"), (24.0, "day"), (68.0, "red"), (STORM, "storm"), (ASH, "ash"), (GOD, "blood")]
WEATHER = dict(
    blaze=[(60, 0), (72, 0.5), (105, 1.0), (STORM, 0.15), (ASH, 0.3), (170, 0.2), (GOD, 1.0)],
    smoke=[(62, 0), (82, 0.35), (STORM, 0.5), (114, 0.15), (ASH, 0.55), (GOD, 0.45)],
    rain=[(111, 0), (115, 0.8), (STRIKE, 1.0), (ASH, 0)],
    ash=[(ASH, 0.0), (146, 0.8), (GOD, 0.5), (182, 0.25)],
    embers=[(64, 0), (82, 0.5), (STORM, 0.3), (112, 0), (GOD, 0.8)],
    strikes=BOLTS, birds=BIRDS, shots=GUN)

# what Distor does when: carrying the rifle, aiming (around each shot), then a god
POSES = [(0.0, "carry")]
for g in GUN:
    POSES += [(g - 0.45, "aim"), (g + 1.1, "carry")]
POSES.append((ASH, "god"))
POSES = sorted(POSES)


def god_level(t):
    """How far gone he is: nothing until the strike, glowing in the ash, fully in the finale."""
    if t < ASH:
        return 0.0
    if t < GOD:
        return 0.25 + 0.5 * (t - ASH) / (GOD - ASH)
    return 1.0


class Empty:
    """No one in the picture (the land alone)."""
    j = {"credit": "Drawing: Ben"}

    def draw(self, frame, t):
        pass


class Rising:
    """A player coming up from below the frame between t0 and t1 (eased)."""

    def __init__(self, p, t0, t1, dy):
        self.p, self.t0, self.t1, self.dy, self.top = p, t0, t1, dy, p.top

    def __getattr__(self, k):
        return getattr(self.p, k)

    def draw(self, frame, t):
        u = float(np.clip((t - self.t0) / (self.t1 - self.t0), 0, 1))
        u = u * u * (3 - 2 * u)
        self.p.top = self.top + self.dy * (1 - u)
        self.p.draw(frame, t)


class HuntShow(Show):
    def hunter(self, scale, top, cx=W / 2, **kw):
        from cast.hunter import Hunter
        kw.setdefault("poses", POSES)
        kw.setdefault("god", god_level)
        return Hunter(self.song, scale=scale, cx=cx, top=top, shots=GUN, sing=[(GOD, END)], **kw)

    def make_cast(self):
        cast = super().make_cast()
        cast["land"] = Empty
        cast["clear"] = Empty
        cast["giant"] = Empty  # the venue draws him (behind the trees)
        cast["giant_land"] = Empty
        cast["hunter"] = lambda: self.hunter(0.6, -140)
        cast["hunter_close"] = lambda: self.hunter(0.9, -330)
        cast["hunter_eye"] = lambda: self.hunter(1.9, -660, cx=W * 0.56)
        cast["walk_hunter"] = lambda: Walking(self.hunter(0.6, -140), self.song)
        cast["hunter_rise"] = lambda: Rising(self.hunter(0.9, -330), self.rise[0] + 0.3, self.rise[1] - 0.8, 700)
        cast["hunter_god"] = lambda: self.hunter(0.9, -330, poses=[(0, "god")])
        return cast

    def full(self, a, b, direction=1):
        """A full-body shot of him walking across a still camera, made for the shot [a, b)."""
        name = f"full_hunter_{len(self._cast)}"
        v = 330.0 * direction  # px/s: a steady walk, coming in from the side
        x0 = W * 0.5 - v * (b - a) * 0.75
        self._cast[name] = lambda: self.hunter(0.25, 40, walk=Path([(a - 0.6, x0 - 0.6 * v), (b + 0.6, x0 + v * (b - a + 0.6))], ramp=0.4))
        return name

    def plan(self):
        bars = self.bars
        shots = []

        def run(a, b, cycle, n=1, opts=None):
            """Shots of n bars from a to b, taking turns in cycle (a cycle entry may be a callable(a, b))."""
            i = int(np.searchsorted(bars, a - 0.05))
            k = 0
            t0 = a
            while t0 < b - 0.05:
                j = min(i + n, len(bars) - 1)
                t1 = min(b, bars[j]) if bars[j] > t0 + 0.2 else b
                who = cycle[k % len(cycle)]
                if callable(who):
                    who = who(t0, t1)
                shots.append((t0, t1, who, dict(opts or {})))
                k += 1
                t0, i = t1, j
        W_ = ("w_guitar", "w_bass")
        D_ = ("d_guitar", "d_bass")
        # the eye: an extreme close-up, the title over it; at the stop he stares (freeze)
        shots.append((0.0, STOP[0], "hunter_eye", dict(zoom=(1.0, 1.06))))
        shots.append((STOP[0], STOP[1], "hunter_eye", dict(tmap=lambda t: STOP[0] - 0.02, zoom=(1.06, 1.12))))
        # nature: the land, the band, him walking in (two bars a shot)
        run(STOP[1], HUNT, ["land", "guitar", D_, lambda a, b: self.full(a, b), "bass", "walk_hunter", W_, "land", "guitar_close",
                            "hunter", "bass_close", lambda a, b: self.full(a, b, -1)], n=2)
        # the hunt: one bar a shot; every gunshot is on him
        run(HUNT, STORM, ["hunter", "land", "guitar", "walk_hunter", W_, "hunter_close", "bass", lambda a, b: self.full(a, b),
                          "guitar_close", D_, "bass_close", "land"], n=1)
        # the storm: the breakdown on him looking up, then the band sings in the rain
        run(STORM, 114.25, ["hunter_close"], n=4, opts=dict(zoom=(1.0, 1.15)))
        run(114.25, STRIKE, ["guitar", "hunter", "guitar_close", W_, "land", "hunter_close", "bass", "guitar_side", D_, "walk_hunter"], n=1)
        shots.append((STRIKE, ASH, "hunter", dict(tmap=lambda t: STRIKE + 0.03)))
        # ash: slow, four bars a shot; he rises (two bars of drum stabs: flash cuts of the giant)
        run(ASH, 170.5, ["land", "clear", "guitar_close", "bass", "hunter_rise", "giant_land"], n=4, opts=dict(zoom=(1.0, 1.08)))
        run(170.5, GOD, ["giant_land", "hunter_god", "giant_land", "hunter_god"], n=1)
        # the god: over the burning woods, the band at his feet
        run(GOD, END, ["giant_land", "hunter_god", ("giant",) + W_, "guitar", "giant_land", "bass", "hunter_god", ("giant",) + D_,
                       "guitar_close", "giant_land", "bass_close", "hunter_god"], n=1)
        shots.append((END, self.song.duration + 1, "hunter_eye", dict(zoom=(1.12, 1.2))))
        out = []
        for a, b, w, o in shots:  # no gaps
            if out:
                a = out[-1][1]
            if b > a + 0.02:
                out.append((a, b, w, o))
        self.rise = next(((a, b) for a, b, w, o in out if w == "hunter_rise"), (ASH, GOD))
        # a gunshot always shows him
        for g in GUN:
            for i, (a, b, w, o) in enumerate(out):
                if a <= g < b and not (isinstance(w, str) and ("hunter" in w)):
                    out[i] = (a, b, "hunter_close" if i % 2 else "hunter", o)
        return out

    def add_moves(self, shots):
        out = super().add_moves([s for s in shots])
        res = []
        for (a, b, w, o) in out:
            o = dict(o)
            if ASH <= a < 170.5 and callable(o.get("focus")):  # no handheld shake on the slow ash shots
                o.pop("focus")
            if any(a <= g < a + 0.2 for g in GUN):  # a gunshot right at the cut: no whip pan hiding it
                o.pop("whip", None)
            res.append((a, b, w, o))
        return res

    def post(self, img, t):
        if self.logo is not None and 0.4 < t < 4.2:
            k = min(1.0, (t - 0.4) / 0.9) * min(1.0, (4.2 - t) / 0.6)
            title_card(img, self.logo, 0.85 * k)
        if t >= END + 0.6:
            k = min(1.0, (t - END - 0.6) / 0.8)
            img[:] = (img * (1 - 0.85 * k)).astype(img.dtype)
            title_card(img, self.logo, k, sub=self.end_lines)
        return img


def giant():
    from cast.hunter import Hunter
    return Hunter(SONG, scale=0.62, cx=W / 2, top=-30, poses=[(0, "god")], god=1.0, sing=[(GOD, END)])


show = HuntShow(SONG, BAND, name="hunt", script=__file__, logo=ben_logo(),
                venue=Wilds(MOODS, WEATHER, giant=giant), style3d="ben", closeups=0, moves=True,
                end_lines=["music: ...", "drawings: Ben"])
prod = show.production()

if __name__ == "__main__":
    prod.main(sys.argv)
