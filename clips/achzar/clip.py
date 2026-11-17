"""Clip 'achzar': Hoodska, "Tsayad Achzar" (the cruel hunter), round 2.

The story: the band's music keeps the forest alive. The hunter speaks (the intro), comes, tracks
them through his scope, fires warning shots, then hits the bassist, who falls. The blonde sings
alone and her song calls the storm: now the crosshair is on him, and lightning strikes him. In the
ash the bassist comes back, and the hunter rises as a god, a giant over the burning woods. The band
plays against him until he crumbles to ash; dawn comes back; the book closes.

Each part tries other camera work (captioned); comic panels and lettering come back throughout:

  1  0:00-0:40  the camera rises and falls   a crane down from the sky, a push through the trees, he walks at us
  2  0:40-1:04  through the scope            his view: the crosshair on the band; the bassist is hit
  3  1:04-1:23  comic pages                  the aftermath, page by page; the camera dives into a panel
  4  1:23-1:50  low, tilted, running         looking up at him, Dutch angles, a handheld chase
  5  1:50-2:25  the hunter is hunted         a red crosshair on him, pages, the storm; lightning hits
  6  2:25-2:58  showdown in silhouette       the bassist rises, the god, the band against him; he crumbles

  python3 clips/achzar/clip.py stills 5 12 50 61.5 70 90 120 150 165
  python3 clips/achzar/clip.py video                -> out/achzar.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
import numpy as np  # noqa: E402

import story  # noqa: E402
from motion import Path  # noqa: E402
from production import H, W  # noqa: E402
from scenery import ben_logo, title_card  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402
from venues import Walking  # noqa: E402
from wilds import Wilds  # noqa: E402

SONG = Song.load("achzar")
BAND = [Member("guitar", "blonde", side="blonde_side", sings=True), Member("bass", "bassboy", seed=1)]
GUITAR, BASS = BAND

# ---- the parts (seconds, from the song)
INTRO, P2, P3, P4, P5, STRIKE, P6, END = 9.3, 40.45, 64.0, 83.0, 110.0, 144.544, 144.962, 175.4
HIT = 61.22  # the bassist is hit
RISE = (144.962, 148.77)  # the bassist comes back
CRUMBLE = (164.0, 171.6)  # the god crumbles
PARTS = [(0.0, "1  the camera rises and falls"), (P2, "2  through the scope"), (P3, "3  comic pages"),
         (P4, "4  low, tilted, running"), (P5, "5  the hunter is hunted"), (P6, "6  showdown in silhouette"), (END, "")]
crash = [e["t"] for e in SONG.data["drums"] if e["piece"] == 49]
kicks = np.array([e["t"] for e in SONG.data["drums"] if e["piece"] == 36])
FRENZY = [t for t in crash if P4 < t < 100]
GUN = [40.45, 48.62, HIT, 72.0, 79.5] + FRENZY + [115.0, 122.6]
BOLTS = [(t, None) for t in crash if 125 < t < STRIKE - 1] + [(STRIKE, W / 2)] + [(t, None) for t in crash if 153 < t < CRUMBLE[1]][::3]
BIRDS = [(10.5, (W * 0.2, H * 0.45), 1), (40.45, (W * 0.5, H * 0.6), 1), (48.62, (W * 0.3, H * 0.62), -1), (HIT, (W * 0.65, H * 0.6), 1),
         (84.73, (W * 0.4, H * 0.62), -1)]
MOODS = [(0.0, "night"), (INTRO, "dawn"), (25.0, "day"), (HIT, "red"), (P4, "blood"), (P5, "storm"), (P6, "ash"), (152.3, "blood"),
         (CRUMBLE[1], "dawn")]
WEATHER = dict(
    blaze=[(HIT, 0), (P3, 0.4), (P4, 0.8), (104, 1.0), (P5, 0.15), (P6, 0.3), (152.3, 1.0), (CRUMBLE[1], 0.3), (END, 0.0)],
    smoke=[(HIT, 0), (70, 0.35), (P5, 0.45), (114, 0.15), (P6, 0.55), (152.3, 0.4), (CRUMBLE[1] + 2, 0.1)],
    rain=[(108, 0), (112, 0.8), (STRIKE, 1.0), (P6, 0)],
    ash=[(P6, 0.0), (146, 0.8), (152.3, 0.4), (160, 0.2), (CRUMBLE[0], 0.3), (CRUMBLE[1], 1.0), (END, 0.3)],
    embers=[(HIT, 0), (P4, 0.6), (P5, 0.2), (113, 0), (152.3, 0.8), (CRUMBLE[1], 0.2)],
    silhouette=[(152.0, 0.0), (152.6, 1.0), (CRUMBLE[1] - 0.5, 1.0), (CRUMBLE[1] + 1.0, 0.0)],
    dissolve=[(CRUMBLE[0], 0.0), (CRUMBLE[1], 1.02)],
    strikes=BOLTS, birds=BIRDS, shots=GUN)

POSES = [(0.0, "carry")]
for g in GUN:
    POSES += [(g - 0.45, "aim"), (g + 1.1, "carry")]
POSES = sorted(POSES + [(P6, "god")])
SING = [(0.0, INTRO), (153.0, CRUMBLE[0])]  # he speaks the intro; he sings as the god
RED = (0.1, 0.1, 0.95)


def god_level(t):
    if t < P6:
        return 0.0
    return 0.3 + 0.7 * min(1.0, (t - P6) / (152.3 - P6))


class AchzarShow(Show):
    def hunter(self, scale, top, cx=W / 2, **kw):
        from cast.hunter import Hunter
        kw.setdefault("poses", POSES)
        kw.setdefault("god", god_level)
        return Hunter(self.song, scale=scale, cx=cx, top=top, shots=GUN, sing=SING, **kw)

    def make_cast(self):
        cast = super().make_cast()
        for n in ("land", "clear", "giant", "giant_land"):
            cast[n] = story.Empty
        cast["hunter"] = lambda: self.hunter(0.6, -140)
        cast["hunter_close"] = lambda: self.hunter(0.9, -330)
        cast["hunter_eye"] = lambda: self.hunter(1.9, -660, cx=W * 0.56)
        cast["walk_hunter"] = lambda: Walking(self.hunter(0.6, -140), self.song)
        cast["full_low_hunter"] = lambda: self.hunter(0.42, 150, aura=False)  # looking up at him: big, low, a lot of sky
        cast["hunter_rise"] = lambda: story.Rising(self.hunter(0.6, -140), RISE[1] + 0.3, 152.0, 600)
        cast["bass_hit"] = lambda: story.Falling(self.bust(BASS, "solo"), HIT + 0.04, dur=1.1)
        cast["bass_rise"] = lambda: story.Rising(self.bust(BASS, "solo"), RISE[0] + 0.3, RISE[1] - 0.4, 650)
        return cast

    def toward(self, a, b, z=(1.0, 1.4)):
        """He walks at us (full body, legs stepping), the camera pushing in on him."""
        name = f"full_hunter_{len(self._cast)}"
        walk = Path([(a - 0.6, 0.0), (b + 0.6, 330.0 * (b - a + 1.2))], ramp=0.4)
        self._cast[name] = lambda: self.hunter(0.25, 40, aura=False, toward=True, walk=walk)
        return name, dict(zoom=z, focus=(W / 2, H * 0.4), drift=0.0)

    def scope(self, shot, target, color=(0.05, 0.05, 0.05), settle=None, mag=1.8):
        name = f"panel_scope_{len(self._cast)}"
        seed = len(self._cast)
        self._cast[name] = lambda: story.Scope(sub_render, shot, target, mag=mag, color=color, seed=seed, settle=settle)
        return name

    def page(self, a, b, layout, shots, dive=None):
        g = 26
        L = {"halves": [(g, g, W // 2 - g * 3 // 2, H - 2 * g), (W // 2 + g // 2, g, W // 2 - g * 3 // 2, H - 2 * g)],
             "big_left": [(g, g, int(W * 0.58), H - 2 * g), (int(W * 0.58) + 2 * g, g, W - int(W * 0.58) - 3 * g, H // 2 - g * 3 // 2),
                          (int(W * 0.58) + 2 * g, H // 2 + g // 2, W - int(W * 0.58) - 3 * g, H // 2 - g * 3 // 2)],
             "strip": [(g, g, W - 2 * g, int(H * 0.42)), (g, int(H * 0.42) + 2 * g, W // 2 - g * 3 // 2, H - int(H * 0.42) - 3 * g),
                       (W // 2 + g // 2, int(H * 0.42) + 2 * g, W // 2 - g * 3 // 2, H - int(H * 0.42) - 3 * g)],
             "thirds": [(g, g, W // 3 - g * 4 // 3, H - 2 * g), (W // 3 + g // 3, g, W // 3 - g * 4 // 3, H - 2 * g),
                        (2 * W // 3 - g // 3 + g, g, W // 3 - g * 4 // 3, H - 2 * g)]}[layout]
        name = f"panel_{len(self._cast)}"
        spec = [(s, r, 0.12) for s, r in zip(shots, L)]
        self._cast[name] = lambda: story.Panels(sub_render, spec, a, b, dive=dive)
        return name, dict(zoom=(1.0, 1.0), drift=0.0, panel=True)

    def plan(self):
        bars = self.bars
        cams = self.venue.cams
        sh = []

        def add(a, b, who, **o):
            sh.append((a, b, who, o))

        def at(t, n):  # the bar line n bars after the one at t
            i = int(np.searchsorted(bars, t - 0.05))
            return bars[min(i + n, len(bars) - 1)]
        # ---- 1: the camera rises and falls
        add(0.0, INTRO, "hunter_eye", zoom=(1.0, 1.12))
        t = INTRO
        b = at(t, 4)
        add(t, b, "land", drift=0.0, zoom=(1.0, 1.0))  # a crane down from the sky into the woods
        cams.append((t, b, dict(crane=(1150.0, 0.0))))
        t, b = b, at(b, 2)
        add(t, b, "guitar")
        t, b = b, at(b, 2)
        add(t, b, "bass")
        t, b = b, at(b, 2)
        add(t, b, "land", drift=0.0, zoom=(1.0, 1.0))  # pushing in through the trees: the near ones rush past
        cams.append((t, b, dict(dolly=(1.0, 1.9), horizon=60.0)))
        t, b = b, at(b, 2)
        n, o = self.toward(t, b, (1.0, 1.25))
        add(t, b, n, **o)
        t, b = b, at(b, 1)
        add(t, b, ("d_guitar", "d_bass"))
        t, b = b, at(b, 2)
        n, o = self.toward(t, b, (1.3, 1.75))
        add(t, b, n, **o)
        add(b, P2, "hunter_close", zoom=(1.0, 1.08))
        # ---- 2: through the scope: warning shots, then the bassist
        add(P2, at(P2, 1), "hunter")
        t = at(P2, 1)
        add(t, at(t, 2), "land", drift=0.0)
        t = at(t, 2)
        b = at(t, 2)
        add(t, b, self.scope("guitar", (W / 2, H * 0.32), settle=48.6))
        t, b = b, at(b, 2)
        add(t, b, "guitar_close", zoom=(1.0, 1.1))
        t, b = b, at(b, 2)
        add(t, b, self.scope("bass", (W / 2, H * 0.3), mag=1.6))
        t, b = b, at(b, 1)
        add(t, b, "hunter_close", zoom=(1.05, 1.2), drift=0.0)
        add(b, HIT, self.scope("bass", (W / 2, H * 0.3), mag=2.3, settle=HIT))
        add(HIT, HIT + 1.5, "bass_hit", drift=0.0)
        add(HIT + 1.5, P3, "hunter_close", zoom=(1.0, 1.06))
        # ---- 3: comic pages (two bars a page; the last dives into a panel)
        pages = [("halves", ["guitar_close", "clear"]), ("big_left", ["w_guitar", "hunter_close", "land"]),
                 ("strip", ["land", "hunter", "guitar"]), ("thirds", ["walk_hunter", "guitar_close", "hunter_close"]),
                 ("halves", ["hunter", "guitar"])]
        t = P3
        for i, (lay, shots) in enumerate(pages):
            b = at(t, 2) if i < len(pages) - 1 else P4
            n, o = self.page(t, b, lay, shots, dive=0 if i == len(pages) - 1 else None)
            add(t, b, n, **o)
            t = b
        # ---- 4: low, tilted, running: the frenzy (one bar a shot)
        cyc = [("full_low_hunter", dict(tilt=-11.0, zoom=(1.2, 1.26))), ("walk_guitar", dict(roll=2.2, zoom=(1.1, 1.1))),
               ("hunter_close", dict(tilt=12.0, zoom=(1.22, 1.28))), ("land", dict(drift=0.0)), ("walk_hunter", dict(roll=2.2, zoom=(1.1, 1.1))),
               ("full_low_hunter", dict(tilt=10.0, zoom=(1.2, 1.28))), ("guitar_close", dict(tilt=-9.0, zoom=(1.18, 1.22))),
               ("walk_guitar_close", dict(roll=2.5, zoom=(1.1, 1.1)))]
        t, k = P4, 0
        while t < P5 - 0.05:
            b = min(at(t, 1), P5)
            who, o = cyc[k % len(cyc)]
            if who == "land":
                cams.append((t, b, dict(dolly=(1.0, 1.5), speed=2.5)))
            if who.startswith("walk"):
                cams.append((t, b, dict(speed=2.6)))
            if who.startswith("full_low"):
                cams.append((t, b, dict(horizon=380.0)))
            add(t, b, who, **dict(o, punch=(kicks, 0.04)))
            t, k = b, k + 1
        # ---- 5: the hunter is hunted (two bars a shot): the storm's red crosshair on him
        seq = [("guitar", None), ("scope", "hunter"), ("page", ("halves", ["hunter_close", "guitar_close"])), ("land", None),
               ("scope", "walk_hunter"), ("guitar_close", None), ("page", ("strip", ["land", "hunter", "guitar"])), ("hunter_close", None),
               ("scope", "hunter_close")]
        t, k = P5, 0
        while t < STRIKE - 0.05:
            b = min(at(t, 2), STRIKE)
            kind, arg = seq[k % len(seq)]
            if b >= STRIKE - 0.05:  # the last: on him, the crosshair closing in
                add(t, b, self.scope("hunter", (W / 2, H * 0.3), color=RED, settle=STRIKE, mag=1.7))
            elif kind == "scope":
                add(t, b, self.scope(arg, (W / 2, H * 0.32), color=RED, mag=1.5))
            elif kind == "page":
                n, o = self.page(t, b, arg[0], arg[1])
                add(t, b, n, **o)
            else:
                add(t, b, kind)
            t, k = b, k + 1
        add(STRIKE, P6, "hunter", tmap=lambda t: STRIKE + 0.03)
        # ---- 6: showdown in silhouette; he crumbles; dawn
        add(RISE[0], RISE[1], "bass_rise", zoom=(1.0, 1.08))
        add(RISE[1], 152.3, "hunter_rise", zoom=(1.0, 1.1))
        cyc = ["giant_land", ("giant", "w_guitar", "w_bass"), "guitar", "giant_land", "bass", ("giant", "d_guitar", "d_bass"),
               "giant_land", "w_guitar"]
        t, k = 152.3, 0
        while t < CRUMBLE[1] - 0.05:
            b = min(at(t, 1), CRUMBLE[1])
            who = cyc[k % len(cyc)] if t < CRUMBLE[0] else (("giant", "w_guitar", "w_bass") if k % 2 else "giant_land")
            o = dict(roll=1.0) if isinstance(who, str) and not who.startswith("giant") else {}
            add(t, b, who, **o)
            t, k = b, k + 1
        add(CRUMBLE[1], END, ("w_guitar", "w_bass"), zoom=(1.0, 1.1), drift=0.0)
        add(END, self.song.duration + 1, ("w_guitar", "w_bass"), zoom=(1.1, 1.14), drift=0.0)
        out = []
        for a, b, w, o in sh:
            if out:
                a = out[-1][1]
            if b > a + 0.02:
                out.append((a, b, w, o))
        return out

    def add_moves(self, shots):
        keep = {i: o for i, (a, b, w, o) in enumerate(shots) if o.get("panel") or "drift" in o or "tilt" in o or "roll" in o}
        out = []
        for i, (a, b, w, o) in enumerate(super().add_moves(shots)):
            o = dict(keep.get(i, o))
            o.pop("panel", None)
            if any(a <= g < a + 0.2 for g in GUN) or a in (HIT, STRIKE):  # a hit at the cut: no whip hiding it
                o.pop("whip", None)
            out.append((a, b, w, o))
        return out

    def post(self, img, t):
        if self.logo is not None and 0.6 < t < INTRO - 0.5:
            k = min(1.0, (t - 0.6) / 0.9) * min(1.0, (INTRO - 0.5 - t) / 0.8)
            title_card(img, self.logo, 0.8 * k)
        for i, g in enumerate(GUN):  # comic lettering on the shots
            if g == HIT:
                story.sfx(img, "BLAM!", t - g, at=(0.3, 0.28), size=190, angle=6, dur=1.1)
            elif g <= t < g + 0.8 and (i % 2 == 0 or g < P4):
                story.sfx(img, "BANG!", t - g, at=(0.7 if i % 3 else 0.28, 0.26 if i % 2 else 0.7), angle=-8 if i % 2 else 7)
        story.sfx(img, "KRAKA-BOOM!", t - STRIKE, at=(0.5, 0.75), size=170, angle=-4, fill=(255, 255, 255), burst=(30, 30, 30), dur=1.4)
        if t >= END:
            story.close_page(img, (t - END - 0.3) / 1.0)
        if t >= END + 1.4:
            k = min(1.0, (t - END - 1.4) / 0.6)
            title_card(img, self.logo, k, sub=self.end_lines)
        cur = [p for p in PARTS if p[0] <= t][-1]
        if cur[1] and t - cur[0] < 6.0 and t > 1.0:  # which experiment, for the first seconds of each part
            story.label(img, cur[1], alpha=0.85 * min(1.0, (6.0 - (t - cur[0])) / 0.8))
        return img


def giant():
    from cast.hunter import Hunter
    return Hunter(SONG, scale=0.62, cx=W / 2, top=-30, poses=[(0, "god")], god=1.0, sing=SING)


show = AchzarShow(SONG, BAND, name="achzar", script=__file__, logo=ben_logo(),
                  venue=Wilds(MOODS, WEATHER, giant=giant), style3d="ben", closeups=0, moves=True,
                  end_lines=['music: Hoodska, "Tsayad Achzar"', "drawings: Ben"])
prod = show.production()


def sub_render(shot, t):
    """One shot's picture at t, uncropped (for comic panels and the scope)."""
    wide = isinstance(shot, tuple)
    fr = prod.stage.frame(t, wide, show.cue(t, shot))
    show.backdrop(fr, t, shot)
    for w in prod.who(shot):
        w.draw(fr, t)
    show.fore(fr, t, shot)
    return fr


if __name__ == "__main__":
    prod.main(sys.argv)
