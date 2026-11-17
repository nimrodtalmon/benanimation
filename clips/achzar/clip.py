"""Clip 'achzar': Hoodska, "Tsayad Achzar" (the cruel hunter). The hunt story (clips/hunt) told over
this song, with a different way of filming in each part, to see what works (a caption names each):

  1  0:00-0:40  classic cuts      the eye (he speaks the intro), nature, he walks in
  2  0:40-1:04  long takes        the hunt starts: slow pushes and pans, few cuts
  3  1:04-1:23  comic panels      stalking: several shots at once in frames on paper
  4  1:23-1:50  beat cuts         the frenzy: a cut every two beats, kicks on the hits
  5  1:50-2:25  noir, on twos     the storm: a two-ink print, drawn on twos; lightning hits him
  6  2:25-2:58  silhouettes, tape the god: black cut-outs against the burning sky, worn tape

  python3 clips/achzar/clip.py stills 5 30 50 70 90 120 160
  python3 clips/achzar/clip.py video                -> out/achzar.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
import numpy as np  # noqa: E402

import story  # noqa: E402
from motion import Path  # noqa: E402
from production import H, HandDrawn, Looks, TVStatic, W  # noqa: E402
from scenery import ben_logo, title_card  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402
from venues import Walking  # noqa: E402
from wilds import Wilds  # noqa: E402

SONG = Song.load("achzar")
BAND = [Member("guitar", "blonde", side="blonde_side", sings=True), Member("bass", "bassboy", seed=1)]

# ---- the parts (seconds, from the song)
INTRO, P2, P3, P4, P5, STRIKE, P6, END = 9.3, 40.45, 64.0, 83.0, 110.0, 144.544, 144.962, 175.4
PARTS = [(0.0, "1  classic cuts"), (P2, "2  long takes"), (P3, "3  comic panels"), (P4, "4  beat cuts"),
         (P5, "5  noir, on twos"), (P6, "6  silhouettes, worn tape"), (END, "")]
crash = [e["t"] for e in SONG.data["drums"] if e["piece"] == 49]
kicks = np.array([e["t"] for e in SONG.data["drums"] if e["piece"] == 36])
GUN = [40.45, 44.6, 48.62, 61.22, 72.0, 79.5] + [t for t in crash if P4 < t < 100] + [115.0, 122.6]
BOLTS = [(t, None) for t in crash if 125 < t < STRIKE - 1] + [(STRIKE, W / 2)] + [(t, None) for t in crash if 153 < t < END][::3]
BIRDS = [(40.45, (W * 0.5, H * 0.6), 1), (48.62, (W * 0.3, H * 0.62), -1), (61.22, (W * 0.65, H * 0.6), 1), (84.73, (W * 0.4, H * 0.62), -1)]
MOODS = [(0.0, "night"), (INTRO, "dawn"), (25.0, "day"), (55.0, "red"), (P4, "blood"), (P5, "storm"), (P6, "ash"), (153.0, "blood")]
WEATHER = dict(
    blaze=[(52, 0), (62, 0.4), (P4, 0.8), (104, 1.0), (P5, 0.15), (P6, 0.3), (153, 1.0)],
    smoke=[(54, 0), (70, 0.35), (P5, 0.45), (114, 0.15), (P6, 0.55), (153, 0.4)],
    rain=[(108, 0), (112, 0.8), (STRIKE, 1.0), (P6, 0)],
    ash=[(P6, 0.0), (146, 0.8), (153, 0.4), (160, 0.2)],
    embers=[(55, 0), (P4, 0.6), (P5, 0.2), (113, 0), (153, 0.8)],
    silhouette=[(P6, 0.0), (P6 + 0.8, 1.0), (END, 1.0), (END + 0.3, 0.0)],
    strikes=BOLTS, birds=BIRDS, shots=GUN)

POSES = [(0.0, "carry")]
for g in GUN:
    POSES += [(g - 0.45, "aim"), (g + 1.1, "carry")]
POSES = sorted(POSES + [(P6, "god")])
SING = [(0.0, INTRO), (153.0, END)]  # he speaks the intro; he sings in the finale


def god_level(t):
    if t < P6:
        return 0.0
    return 0.3 + 0.7 * min(1.0, (t - P6) / (153.0 - P6))


TAPE = Looks(HandDrawn(), TVStatic(0.14, vhs=0.9))
NOIR = story.Duotone()


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
        cast["hunter_rise"] = lambda: story.Rising(self.hunter(0.6, -140), P6 + 0.5, 152.0, 600)
        cast["hunter_god"] = lambda: self.hunter(0.9, -330, poses=[(0, "god")])
        return cast

    def full(self, a, b, direction=1):
        name = f"full_hunter_{len(self._cast)}"
        v = 330.0 * direction
        x0 = W * 0.5 - v * (b - a) * 0.75
        self._cast[name] = lambda: self.hunter(0.25, 40, aura=False, walk=Path([(a - 0.6, x0 - 0.6 * v), (b + 0.6, x0 + v * (b - a + 0.6))], ramp=0.4))
        return name

    def panels(self, a, b, layout, shots):
        """A comic-panel shot: shots laid out by layout ("halves", "big_left", "strip")."""
        g = 26
        L = {"halves": [(g, g, W // 2 - g * 3 // 2, H - 2 * g), (W // 2 + g // 2, g, W // 2 - g * 3 // 2, H - 2 * g)],
             "big_left": [(g, g, int(W * 0.58), H - 2 * g), (int(W * 0.58) + 2 * g, g, W - int(W * 0.58) - 3 * g, H // 2 - g * 3 // 2),
                          (int(W * 0.58) + 2 * g, H // 2 + g // 2, W - int(W * 0.58) - 3 * g, H // 2 - g * 3 // 2)],
             "strip": [(g, g, W - 2 * g, int(H * 0.42)), (g, int(H * 0.42) + 2 * g, W // 2 - g * 3 // 2, H - int(H * 0.42) - 3 * g),
                       (W // 2 + g // 2, int(H * 0.42) + 2 * g, W // 2 - g * 3 // 2, H - int(H * 0.42) - 3 * g)]}[layout]
        name = f"panel_{len(self._cast)}"
        spec = [(s, r, 0.12) for s, r in zip(shots, L)]
        self._cast[name] = lambda: story.Panels(sub_render, spec, a, b)
        return name

    def plan(self):
        bars, beats = self.bars, list(self.song.data["beats"]) + [self.song.duration + 1]
        W_, D_ = ("w_guitar", "w_bass"), ("d_guitar", "d_bass")
        sh = []
        # 1: classic: the eye (he speaks), then two bars a shot
        sh.append((0.0, INTRO, "hunter_eye", dict(zoom=(1.0, 1.12))))
        sh += story.cuts(bars, INTRO, P2, ["land", "guitar", D_, lambda a, b: self.full(a, b), "bass", "walk_hunter", W_, "land",
                                           "guitar_close", "hunter", "bass_close", lambda a, b: self.full(a, b, -1)], n=2)
        # 2: long takes: four bars a shot, a slow push and pan, no shake
        sh += story.cuts(bars, P2, P3, ["hunter", lambda a, b: self.full(a, b), "guitar", "hunter_close"], n=4,
                         opts=lambda a, b: dict(zoom=(1.02, 1.16), focus=story.slow_pan(a, b, 140), drift=0.0, long=True))
        # 3: comic panels: two bars a page
        pages = [("halves", ["hunter", "guitar"]), ("big_left", ["land", "hunter_close", "bass"]), ("strip", [W_, "guitar_close", "hunter"]),
                 ("big_left", ["walk_hunter", "bass_close", "guitar"]), ("halves", ["hunter_close", D_])]
        k = [0]

        def page(a, b):
            lay, shots = pages[k[0] % len(pages)]
            k[0] += 1
            return self.panels(a, b, lay, shots)
        sh += story.cuts(bars, P3, P4, [page], n=2, opts=dict(zoom=(1.0, 1.0), drift=0.0, panel=True))
        # 4: beat cuts: two beats a shot, a kick on every bass drum, whips every fourth
        cyc = ["hunter", "guitar", "land", "hunter_close", "bass", W_, "walk_hunter", "guitar_close", "hunter", D_, "bass_close", "land"]
        cnt = [0]

        def fast(a, b):
            cnt[0] += 1
            return dict(zoom=(1.05, 1.1), drift=0.0, punch=(kicks, 0.05), **({"whip": 1 if cnt[0] % 8 == 0 else -1} if cnt[0] % 4 == 0 else {}))
        sh += story.beat_cuts(beats, P4, P5, cyc, n=2, opts=fast)
        # 5: noir on twos: one bar a shot; the strike freezes on the break
        sh += story.cuts(bars, P5, STRIKE, ["hunter_close", "guitar", "land", "hunter", W_, "guitar_close", "bass", "walk_hunter", "guitar_side", D_],
                         n=1, opts=dict(look=NOIR, twos=True))
        sh.append((STRIKE, P6, "hunter", dict(tmap=lambda t: STRIKE + 0.03, look=NOIR)))
        # 6: silhouettes on worn tape: he rises, then the god over the burning woods, the band at his feet
        tape = dict(look=TAPE, roll=1.2)
        sh.append((P6, 152.3, "hunter_rise", dict(tape, zoom=(1.0, 1.1))))
        sh += story.cuts(bars, 152.3, END, ["giant_land", lambda a, b: self.full(a, b), ("giant",) + W_, "guitar", "giant_land", "bass",
                                           lambda a, b: self.full(a, b, -1), ("giant",) + D_, "w_guitar", "giant_land", "w_bass", "hunter_eye"], n=1, opts=tape)
        sh.append((END, self.song.duration + 1, "hunter_eye", dict(zoom=(1.12, 1.2))))
        out = []
        for a, b, w, o in sh:
            if out:
                a = out[-1][1]
            if b > a + 0.02:
                out.append((a, b, w, o))
        for g in GUN:  # a gunshot always shows him (in panels he is there already)
            for i, (a, b, w, o) in enumerate(out):
                if a <= g < b and not (isinstance(w, str) and ("hunter" in w or w.startswith("panel"))):
                    out[i] = (a, b, "hunter_close" if i % 2 else "hunter", o)
        return out

    def add_moves(self, shots):
        keep = {i: o for i, (a, b, w, o) in enumerate(shots) if o.get("long") or o.get("panel")}  # their own camera
        out = []
        for i, (a, b, w, o) in enumerate(super().add_moves(shots)):
            o = dict(keep.get(i, o))
            o.pop("long", None)
            o.pop("panel", None)
            if any(a <= g < a + 0.2 for g in GUN):  # a gunshot at the cut: no whip hiding it
                o.pop("whip", None)
            out.append((a, b, w, o))
        return out

    def post(self, img, t):
        if self.logo is not None and 0.6 < t < INTRO - 0.5:
            k = min(1.0, (t - 0.6) / 0.9) * min(1.0, (INTRO - 0.5 - t) / 0.8)
            title_card(img, self.logo, 0.8 * k)
        if t >= END + 0.6:
            k = min(1.0, (t - END - 0.6) / 0.8)
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
    """One shot's picture at t, uncropped (for comic panels)."""
    wide = isinstance(shot, tuple)
    fr = prod.stage.frame(t, wide, show.cue(t, shot))
    show.backdrop(fr, t, shot)
    for w in prod.who(shot):
        w.draw(fr, t)
    show.fore(fr, t, shot)
    return fr


if __name__ == "__main__":
    prod.main(sys.argv)
