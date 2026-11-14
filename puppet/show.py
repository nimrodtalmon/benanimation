"""A band show from a song and a band, with nothing written per clip: the cast in every shot size,
the shot list, the light cues, the breaks, flashes, titles and the end card all come from the song
(puppet/new_song.py) and the band's rigs (public/rigs/*.json).

  from show import Member, Show
  band = [Member("drums", "distor"), Member("guitar", "blonde", sings=True, side="blonde_side"),
          Member("bass", "bassboy")]
  show = Show(Song.load("dr1"), band, name="distorted", script=__file__, logo=ben_logo(),
              end_lines=["music: Distorted Reality", "drawings: Ben"])
  prod = show.production()          # a Production: .main(argv) renders stills/segments/video

How the show is built (the Clint Eastwood grammar, see docs/TOOLKIT.md):
- intro (until the band comes in, the first loud bar): the players who play in it, alone in one
  spotlight, two bars per shot, slow pushes; the logo fades in over it.
- the band comes in: a white flash, the lights and the logo sign on the back wall come on, the
  shot is the drummer (or the first member); then one bar per shot, cycling through the players
  (solo, close-up), the wide shot and the front pair; while someone sings, every other shot is
  theirs; now and then a side view.
- breaks (stop-time holes): the picture freezes on the hit, the lights drop, the sign glows red,
  a flash and a shout when the band comes back.
- twice in the middle, a bar played forward/backward (the tape trick); flash cuts in the last
  loud stretch; the camera moves more as the song goes on.
- after the last hit: the lights fade out and the logo comes back with the credits.

Shot sizes are set from each character's head size in its drawing (so new characters need no
tuning): solo, close, wide and front-pair framings put the head at fixed sizes and heights.
"""
import math
import os
from dataclasses import dataclass

import numpy as np

import rig
from groove import Groove
from production import H, W, HandDrawn, Looks, Production, Stage, TVStatic
from scenery import Haze, Sign, title_card


@dataclass
class Member:
    role: str  # "drums", "guitar", "bass" (drums: at the kit; others: an instrument from the rig's "play")
    rig: str  # public/parts/<rig>
    sings: bool = False  # the lead voice (mouth from the vocal track, a mic)
    side: str = None  # a rig of the same character from the side (profile shots), if there is one
    seed: int = 0


# framing for bust players: (head height on screen, head top y) per shot size
FRAMES = dict(solo=(360, 125), close=(545, 60), wide=(225, 420), duo=(297, 215))
KIT = dict(solo=dict(scale=0.6, top=20), close=dict(scale=0.85, top=-60), wide=dict(scale=0.36, top=150))


def head_box(j):
    """Head height and head top in the body drawing (from the face landmarks)."""
    fx, fy, frx, fry = j["head"]["face"][:4]
    return 2 * fry, fy - fry


class Show:
    def __init__(self, song, band, name, script, logo=None, end_lines=(), marker=0.25, tv=(0.05, 0.25), head=0.45,
                 style="mix", groove_drums=0.6, flash_cuts=True, tape_trick=2, kit="classic", kit_size=1.15):
        """kit: a 3D kit (public/kits/<kit>.json, puppet/kit3d.py) or None for the code-drawn one."""
        self.song, self.band, self.name, self.script = song, band, name, os.path.abspath(script)
        self.logo, self.end_lines = logo, list(end_lines)
        self.marker, self.tv, self.head, self.style = marker, tv, head, style
        self.groove_drums, self.flash_cuts, self.tape_trick = groove_drums, flash_cuts, tape_trick
        self.kit_name, self.kit_size = kit, kit_size
        d = song.data
        self.holes = [tuple(h) for h in d.get("holes", [])]
        self.drums = [m for m in band if m.role == "drums"]
        self.front = [m for m in band if m.role != "drums"]
        self.singers = [m for m in band if m.sings]
        g = Groove(song)
        self.energy = g.energy
        bars = d["beats"][::4] if d.get("beats") else list(np.arange(0, song.duration, song.bar))
        self.bars = list(bars) + [song.duration + 1]
        loud = [b for b in bars if g.e(b + 0.5 * song.bar) > 0.55]
        self.band_in = loud[0] if loud else 0.0
        hits = [e["t"] for k in ("drums", "bass", "guitar", "vocal") for e in d.get(k, [])]
        self.end = (max(hits) + 0.25) if hits else song.duration
        self.stops = [h for h in self.holes if h[0] > self.band_in + 1]
        self.shouts = [self.band_in] + [b for a, b in self.stops]
        self.crash = np.array([e["t"] for e in d.get("drums", []) if e["piece"] == 49])
        self.voice = song.vocal_drivers() if self.singers else None
        self.sign = Sign(logo, width=820, y=70) if logo is not None else None
        self.haze = Haze()
        self._cast = self.make_cast()
        self.shots = self.plan()

    # ---- the cast in every shot size
    def x_of(self, m, kind):
        """Where a front member stands (screen x) in the wide shot / the front pair."""
        i, n = self.front.index(m), len(self.front)
        if kind == "duo":
            pair = self.front[:2]
            return W * (0.30 if m is pair[0] else 0.72) if m in pair else None
        xs = {1: [0.5], 2: [0.24, 0.77], 3: [0.18, 0.5, 0.82]}.get(n, list(np.linspace(0.15, 0.85, n)))
        return W * xs[i]

    def bust(self, m, kind, x=None):
        from cast.bust_player import BustPlayer
        j = rig.load(m.rig)
        hh, ht = head_box(j)
        size, ytop = FRAMES[kind]
        sc = size / hh
        return BustPlayer(self.song, m.rig, scale=sc, cx=W / 2 if x is None else x, top=int(ytop - ht * sc), head=self.head,
                          seed=3 + 2 * m.seed, style=self.style, shouts=self.shouts, sings=m.sings)

    def kit(self, m, kind):
        from cast.kit_player import KitPlayer
        k = KIT[kind]
        return KitPlayer(self.song, m.rig, scale=k["scale"], top=k["top"], head=self.head, face=True, groove=self.groove_drums,
                         kit=self.kit_name, kit_size=self.kit_size)

    def make_cast(self):
        cast, wide, duo = {}, [], []
        for m in self.band:
            r = m.role
            if r == "drums":
                cast[r] = (lambda m=m: self.kit(m, "solo"))
                cast[r + "_close"] = (lambda m=m: self.kit(m, "close"))
                cast["w_" + r] = (lambda m=m: self.kit(m, "wide"))
            else:
                cast[r] = (lambda m=m: self.bust(m, "solo"))
                cast[r + "_close"] = (lambda m=m: self.bust(m, "close"))
                cast["w_" + r] = (lambda m=m: self.bust(m, "wide", self.x_of(m, "wide")))
                if self.x_of(m, "duo") is not None:
                    cast["d_" + r] = (lambda m=m: self.bust(m, "duo", self.x_of(m, "duo")))
                    duo.append("d_" + r)
            if m.side:
                from cast.portrait import Portrait
                cast[r + "_side"] = (lambda m=m: Portrait(self.song, m.side, head=self.head, seed=m.seed, shouts=self.shouts, sings=m.sings))
            wide.append("w_" + r)
        # the wide shot draws back to front: the drummer first
        self.WIDE = tuple(sorted(wide, key=lambda n: 0 if n == "w_drums" else 1))
        self.DUO = tuple(duo) if len(duo) == 2 else None
        return cast

    # ---- the shot list
    def singing(self, t0, t1):
        if not self.voice:
            return False
        sg = self.voice[2]
        a, b = int(t0 * 30), int(t1 * 30)
        return bool(sg[a:b].mean() > 0.5) if b > a else False

    def playing_in(self, m, t0, t1):
        d = self.song.data
        key = {"guitar": "guitar", "bass": "bass", "drums": "drums"}.get(m.role)
        n = sum(1 for e in d.get(key, []) if t0 <= e["t"] < t1) if key else 0
        if m.sings and self.singing(t0, t1):
            n += 5
        return n > 1

    def plan(self):
        song, spb, bars = self.song, self.song.spb, self.bars
        first = self.drums[0] if self.drums else self.band[0]
        solo = [m.role for m in self.front]
        close = [m.role + "_close" for m in self.front]
        cycle = [first.role]
        if self.WIDE:
            cycle.append(self.WIDE)
        for i in range(max(len(solo), 2)):
            if i < len(solo):
                cycle.append(solo[i])
            if i == 0 and self.drums:
                cycle.append(first.role + "_close")
            if i == 0 and self.DUO:
                cycle.append(self.DUO)
        cycle += close + ([self.WIDE] if self.WIDE else []) + ([self.drums[0].role] if self.drums else []) + ([self.DUO] if self.DUO else [])
        sides = [m.role + "_side" for m in self.band if m.side]
        sing_shots = [m.role + "_close" if k % 2 else m.role for m in self.singers for k in range(2)]
        mid0, mid1 = self.band_in + 0.35 * (self.end - self.band_in), self.band_in + 0.7 * (self.end - self.band_in)
        flash0 = self.band_in + 0.75 * (self.end - self.band_in)
        flash_set = [r for r in close + [first.role + "_close"] + solo]
        shots, i, k, kc, kf, ks, nrock = [], 0, 0, 0, 0, 0, 0
        t0 = 0.0
        while i < len(bars) - 1:
            if bars[i] < self.band_in - 0.3:  # intro: who plays here, two bars a shot, slow pushes
                j = min(i + 2, len(bars) - 1)
                while bars[j] > self.band_in - 0.3 and j > i + 1:
                    j -= 1
                feat = [m for m in self.band if self.playing_in(m, bars[i], bars[j])] or [self.band[0]]
                m = feat[k % len(feat)]
                opts = [m.role, self.WIDE, m.role + "_close"]
                shots.append((t0, bars[j], opts[k % 3] or m.role, dict(zoom=(1.0, 1.04) if k % 2 == 0 else (1.04, 1.0))))
                k += 1
                t0, i = bars[j], j
                continue
            t1 = bars[i + 1]
            last = shots[-1][2] if shots else None
            singer_last = isinstance(last, str) and any(last.startswith(m.role) for m in self.singers)
            if self.singing(t0, t1) and sing_shots and not singer_last:  # while someone sings: every other shot is theirs
                who = sing_shots[ks % len(sing_shots)]
                ks += 1
                if sides and ks % 5 == 4:  # now and then from the side
                    who = next((m.role + "_side" for m in self.singers if m.side), who)
            else:
                who = cycle[kc % len(cycle)]  # the rotation keeps its order (no one is skipped)
                kc += 1
                if sides and kc % 9 == 7 and not self.singers:
                    who = sides[(kc // 9) % len(sides)]
            hs = [h for h in self.stops if t0 - 0.05 <= h[0] < t1]
            if hs:  # a break inside this bar: play up to it, freeze on its first frame
                a_, b_ = hs[0]
                shots.append((t0, a_, who, {}))
                shots.append((a_, min(t1, b_ + 0.02), who, dict(tmap=(lambda a=a_: (lambda t: a + 0.03))())))
                if b_ + 0.02 < t1:
                    shots.append((b_ + 0.02, t1, cycle[kc % len(cycle)], {}))
                    kc += 1
            elif self.flash_cuts and flash0 <= t0 < self.end - 2:  # flash cuts: a half bar, two quarter bars
                q = (t1 - t0) / 4
                for (u0, u1) in ((0, 2), (2, 3), (3, 4)):
                    shots.append((t0 + u0 * q, t0 + u1 * q, flash_set[kf % len(flash_set)], {}))
                    kf += 1
            elif mid0 <= t0 < mid1 and isinstance(who, tuple) and nrock < self.tape_trick:
                shots.append((t0, t1, who, dict(tmap=rock(t0, spb))))
                nrock += 1
            else:
                shots.append((t0, t1, who, {}))
            t0, i = t1, i + 1
        shots = [sh for sh in shots if sh[1] - sh[0] > 0.02]
        out = []
        for n, (a, b, w, o) in enumerate(shots):
            nb = shots[n + 1][0] if n + 1 < len(shots) else b
            a = out[-1][1] if out else 0.0
            if nb > a:
                out.append((a, nb, w, o))
        return out

    # ---- lights, sign, haze, flashes, titles
    def hole_at(self, t):
        for a, b in self.holes:
            if a <= t < b:
                return a, b
        return None

    def spot_x(self, shot, t):
        """The intro spotlight: on whoever plays alone in the intro (their place in this shot)."""
        names = shot if isinstance(shot, tuple) else (shot,)
        for n in names:
            role = n[2:] if n[:2] in ("w_", "d_") else n.replace("_close", "").replace("_side", "")
            m = next((m for m in self.band if m.role == role), None)
            if m and self.playing_in(m, max(0, t - 2), t + 2):
                if n.startswith("w_"):
                    return self.x_of(m, "wide") if m in self.front else W / 2
                if n.startswith("d_"):
                    return self.x_of(m, "duo")
                return W / 2
        return None

    def cue(self, t, shot):
        if t < self.band_in:
            x = self.spot_x(shot, t)
            k = 0.35 if self.hole_at(t) else 1.0
            return dict(beams=0.12, house=0.55 * k, spot=(x, 1.3 * k) if x is not None else None)
        if t >= self.end:
            return dict(beams=0.0, house=max(0.0, 1 - (t - self.end) / 0.6))
        if self.hole_at(t):
            return dict(beams=0.0, house=0.3)
        return dict(beams=1.0, house=1.0)

    def backdrop(self, frame, t, shot):
        if self.sign is None or t < self.band_in - 0.05 or t >= self.end + 0.4:
            return
        on = min(1.0, (t - self.band_in + 0.05) / 0.08)
        bar = int(t / self.song.bar) % 4
        col = (0.30, 0.25, 1.0) if self.hole_at(t) else tuple(0.5 + 0.5 * np.array(Stage.COLORS[bar]))
        self.sign.draw(frame, t, col, 0.8 * on)

    def fore(self, frame, t, shot):
        self.haze.draw(frame, t, 1.4 if t < self.band_in else 0.8)
        flash = 0.0
        for t_on in self.shouts:
            u = t - t_on
            if 0 <= u < 0.4:
                flash = max(flash, math.exp(-u / 0.09))
        if flash > 0 and self.band_in > 0.5:
            frame += 0.35 * flash

    def post(self, img, t):
        if self.logo is None:
            return img
        if t < 4.2 and self.band_in > 4:
            k = min(1.0, max(0.0, (t - 0.4) / 0.9)) * min(1.0, max(0.0, (4.0 - t) / 0.8))
            title_card(img, self.logo, 0.85 * k)
        elif t >= self.end:
            title_card(img, self.logo, min(1.0, (t - self.end) / 0.5), sub=self.end_lines)
        return img

    def production(self):
        b0 = self.band_in
        return Production(self.song, self._cast, self.shots, name=self.name, script=self.script,
                          look=Looks(HandDrawn(), TVStatic(self.tv[0], vhs=self.tv[1])), stage="stage", crowd_wide=False, cue=self.cue,
                          drift=lambda a: 0.015 if a < b0 else (0.05 if a < b0 + 0.35 * (self.end - b0) else (0.065 if a < b0 + 0.7 * (self.end - b0) else 0.09)),
                          parallax=0.55, ghost=0.35, backdrop=self.backdrop, fore=self.fore, post=self.post, marker=self.marker)


def rock(a, spb):
    """The tape trick over one bar: beats 1-2 as played, beat 3 runs backward over beat 2, beat 4
    plays beat 2 forward again (the audio keeps going)."""
    def f(t):
        u = t - a
        if u < 2 * spb:
            return t
        if u < 3 * spb:
            return a + 2 * spb - (u - 2 * spb)
        return a + spb + (u - 3 * spb)
    return f
