"""Where a Show (puppet/show.py) happens. A venue gives the stage (the empty picture), the light cues,
what goes behind the cast (backdrop) and in front of it (fore). Show keeps the cast, the shot list
and the titles, so the same band and song can play anywhere:

  Show(song, band, ..., venue=Club())       # a stage: beams, the logo sign, haze, white flashes
  Show(song, band, ..., venue=Campfire())   # the intro is a walk through the woods at dusk; when
                                            # the band comes in, a fire at night in a clearing

A venue with travel = True gets walking versions of the front players for the intro
("walk_<role>", "walk_<role>_close", the pair "walk_w_<role>"): Walking wraps any bust player.
"""
import math

import numpy as np

from outdoors import Fire, Sky, Sparks, firelight
from production import H, W, Stage
from scenery import Haze, Sign


class Venue:
    travel = False  # walking shots in the intro
    tv = None  # (static, vhs) for this venue, or None: the show's own

    def bind(self, show):
        self.show = show

    def stage(self, song):
        return "stage"

    def cue(self, t, shot):
        return None

    def backdrop(self, frame, t, shot):
        pass

    def fore(self, frame, t, shot):
        pass


class Club(Venue):
    """The stage: a spotlight in the intro, beams and the logo sign when the band comes in, the
    lights drop in the breaks, white flashes on the shouts."""

    def bind(self, show):
        self.show = show
        self.sign = Sign(show.logo, width=820, y=70) if show.logo is not None else None
        self.haze = Haze()

    def cue(self, t, shot):
        s = self.show
        if t < s.band_in:
            x = s.spot_x(shot, t)
            k = 0.35 if s.hole_at(t) else 1.0
            return dict(beams=0.12, house=0.55 * k, spot=(x, 1.3 * k) if x is not None else None)
        if t >= s.end:
            return dict(beams=0.0, house=max(0.0, 1 - (t - s.end) / 0.6))
        if s.hole_at(t):
            return dict(beams=0.0, house=0.3)
        return dict(beams=1.0, house=1.0)

    def backdrop(self, frame, t, shot):
        s = self.show
        if self.sign is None or t < s.band_in - 0.05 or t >= s.end + 0.4:
            return
        on = min(1.0, (t - s.band_in + 0.05) / 0.08)
        bar = int(t / s.song.bar) % 4
        col = (0.30, 0.25, 1.0) if s.hole_at(t) else tuple(0.5 + 0.5 * np.array(Stage.COLORS[bar]))
        self.sign.draw(frame, t, col, 0.8 * on)

    def fore(self, frame, t, shot):
        s = self.show
        self.haze.draw(frame, t, 1.4 if t < s.band_in else 0.8)
        flash = 0.0
        for t_on in s.shouts:
            u = t - t_on
            if 0 <= u < 0.4:
                flash = max(flash, math.exp(-u / 0.09))
        if flash > 0 and s.band_in > 0.5:
            frame += 0.35 * flash


class Walking:
    """Any bust player, walking: one step a beat (the body dips on each footfall and leans into the
    step), the scenery scrolls by behind (the venue's job). Delegates everything else."""

    def __init__(self, player, song, phase=0.0, bob=14, sway=6):
        self.p, self.spb, self.phase = player, song.spb, phase
        self.x0, self.top = player.x0, player.top
        self.bob, self.sway = bob, sway

    def __getattr__(self, k):
        return getattr(self.p, k)

    def draw(self, frame, t):
        u = t / self.spb + self.phase  # one step a beat
        s = self.p.s / 0.5
        step = abs(math.sin(math.pi * u))  # 0 at the footfall
        self.p.top = self.top + self.bob * s * (0.6 - step) - 4 * s * math.exp(-((u % 1.0) / 0.12))
        self.p.x0 = self.x0 + self.sway * s * math.sin(math.pi * u)
        self.p.draw(frame, t)


class Campfire(Venue):
    """Dusk to night in the woods. Intro: the players walk (the hills and pines scroll by, ferns
    pass in front); the light goes from sunset to dusk and a warm glow shows up ahead. The band
    comes in: a clearing, a fire flares up. The fire follows the band: it sinks to embers in the
    breaks and flares on the shouts, throws sparks on the crashes and more in the loudest parts,
    and dies down after the last hit. Firelight from below on everyone."""
    travel = True
    tv = (0.03, 0.15)

    def __init__(self, seed=5, walk_speed=260.0):
        self.seed, self.walk_speed = seed, walk_speed

    def bind(self, show):
        self.show = show
        s, song = show, show.song
        n = int(song.duration * 30) + 60
        tt = np.arange(n) / 30.0
        heat = np.zeros(n, np.float32)
        on = tt >= s.band_in
        heat[on] = 1.0
        for a, b in s.holes:  # the breaks: embers
            if a > s.band_in:
                heat[(tt >= a) & (tt < b + 0.05)] = 0.35
        for a, b in song.data.get("peaks", []):
            heat[(tt >= a) & (tt < b)] *= 1.25
        en = np.asarray(s.energy, np.float32)  # Groove.energy, at 30 fps
        e = en[np.clip((tt * 30).astype(int), 0, len(en) - 1)]
        heat *= 0.85 + 0.3 * e
        for t_on in s.shouts:  # flare-ups
            u = tt - t_on
            heat += np.where((u >= 0) & (u < 2.0), 0.9 * np.exp(-np.clip(u, 0, None) / 0.35), 0)
        after = tt >= s.end
        heat[after] = np.maximum(0.12, heat[after] * np.exp(-(tt[after] - s.end) / 1.2))
        self.heat = np.convolve(heat, np.ones(3) / 3, "same")
        crash = s.crash if len(s.crash) else np.array([])
        self.sparks = Sparks(song.duration + 2, lambda x: self.heat_at(x), list(crash) + [x for x in s.shouts for _ in range(4)], seed=self.seed)
        self.sky = Sky(seed=self.seed)
        self.fire = Fire(seed=self.seed)

    def heat_at(self, t):
        return float(self.heat[min(len(self.heat) - 1, max(0, int(t * 30)))])

    def dusk(self, t):
        """0: sunset, 1: night."""
        b = self.show.band_in
        return float(np.clip(0.15 + 0.55 * t / max(b, 1.0), 0, 0.7)) if t < b else float(min(1.0, 0.8 + (t - b) / 8.0))

    def walking(self, shot):
        names = shot if isinstance(shot, tuple) else (shot,)
        return any(str(n).startswith("walk_") for n in names)

    def stage(self, song):
        return self

    dark, vignette = False, 0.4

    def frame(self, t, wide, cue):
        shot = cue["shot"]
        travel = self.walk_speed * t if self.walking(shot) else 0.0
        return self.sky.frame(self.dusk(t), t, travel, clearing=not self.walking(shot))

    def cue(self, t, shot):
        return dict(shot=shot)

    def fire_spot(self, shot):
        """Where the fire is in this shot: (x, base y, scale); off-screen below for the singles."""
        if isinstance(shot, tuple):
            return W / 2, H * 0.99, 0.8  # the wide shot: in front of the band, in the middle
        if shot.endswith("_side"):
            return W * 0.18, H + 140, 2.2
        if shot.endswith("_close"):
            return W / 2, H + 260, 2.6
        return W / 2, H + 170, 2.0

    def fore(self, frame, t, shot):
        s = self.show
        if self.walking(shot):
            d = self.dusk(t)
            ahead = float(np.clip((t - (s.band_in - 5.0)) / 5.0, 0, 1))  # the fire's glow ahead
            firelight(frame, (W * 1.05, H * 0.75), 0.0, dusk=d, glow=0.5 * ahead, sun=1 - d, t=t)
            self.sky.ferns(frame, t, self.walk_speed * t, d)
            return
        x, y, sc = self.fire_spot(shot)
        h = self.heat_at(t)
        firelight(frame, (x, y), h, dusk=self.dusk(t), t=t)
        self.sparks.draw(frame, t, (x, y), sc)
        if y < H + 50:
            self.fire.draw(frame, t, (x, y), sc, h)
        for t_on in s.shouts[1:]:  # a warm flash with the shout (the first one is the ignition)
            u = t - t_on
            if 0 <= u < 0.4:
                frame += np.array([0.05, 0.16, 0.30], np.float32) * math.exp(-u / 0.09)
        u = t - s.band_in
        if 0 <= u < 0.6:  # the ignition
            frame += np.array([0.10, 0.30, 0.55], np.float32) * math.exp(-u / 0.12)
