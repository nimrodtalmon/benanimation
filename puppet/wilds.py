"""A wild place for story clips (the hunt): woods, far mountains and an open sky (puppet/scenery3d.py
Wilds3D, drawn in the ink style), with the light, the sky and the weather following a story score:

  Wilds(moods=[(t, mood), ...], weather=dict(...), giant=factory)

moods: the light of the moment (crossfaded): MOODS below ("dawn", "day", "red", "storm", "ash",
  "blood", "night"), each a sky (puppet/skies.py) and how the land and the cast are lit.
weather (all optional): rain/ash/embers/smoke/blaze: [(t, amount), ...] keyframes (linear);
  strikes: [(t, x)] lightning; birds: [(t, (x, y), toward)] flocks; shots: [t] gunshot flashes.
giant: a factory for a cast member drawn huge BEHIND the trees in "giant" shots (between the
  far mountains and the near trees).

Which view a shot gets comes from its cast names: "land*" (the land alone, side-on, scrolling),
"walk*" (side-on, scrolling: walking busts), "full*" (side-on, still camera: someone walks
across), "giant*" (side-on, low horizon, the giant behind the trees); anything else: the clearing.
"""
import math

import numpy as np

from production import H, W
from venues import Venue
import weather as wx

MOODS = {
    #        sky       land tint (BGR)       sun   moon  blaze  cast light (BGR)
    "night": ("stars", (0.55, 0.45, 0.42), 0.0, 1.0, 0.0, (0.75, 0.68, 0.66)),
    "dawn": ("dawn", (0.92, 0.92, 1.0), 1.0, 0.0, 0.0, (0.95, 0.97, 1.05)),
    "day": ("day", (1.05, 1.05, 1.0), 1.1, 0.0, 0.0, (1.02, 1.02, 1.0)),
    "red": ("ash", (0.6, 0.68, 0.95), 0.8, 0.0, 0.5, (0.82, 0.88, 1.05)),
    "storm": ("storm", (0.5, 0.5, 0.52), 0.65, 0.3, 0.0, (0.72, 0.72, 0.74)),
    "ash": ("ash", (0.62, 0.64, 0.74), 0.75, 0.2, 0.25, (0.72, 0.74, 0.82)),
    "blood": ("blood", (0.32, 0.36, 0.95), 0.45, 0.0, 1.0, (0.55, 0.6, 1.1)),
}


def key(keys, t, default=0.0):
    """Linear keyframes [(t, v), ...] -> v at t (held before the first and after the last)."""
    if not keys:
        return default
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return keys[0][1]
    if t >= ts[-1]:
        return keys[-1][1]
    i = int(np.searchsorted(ts, t, side="right")) - 1
    (t0, v0), (t1, v1) = keys[i], keys[i + 1]
    return v0 + (v1 - v0) * (t - t0) / max(1e-6, t1 - t0)


class Wilds(Venue):
    tv = (0.03, 0.12)
    dark, vignette = False, 0.4

    def __init__(self, moods, weather=None, giant=None, seed=5, scenery="wilds", walk_speed=240.0, fade=1.5, shift=0.05):
        self.moods, self.wx_spec, self.giant_f = sorted(moods), dict(weather or {}), giant
        self.seed, self.scenery_name, self.walk_speed, self.fade, self.shift = seed, scenery, walk_speed, fade, shift
        self._giant = None

    def bind(self, show):
        from scenery3d import Wilds3D
        from skies import Skies
        self.show = show
        self.scn = Wilds3D(self.scenery_name, shift=self.shift)
        self.skies = Skies(self.seed)
        w = self.wx_spec
        self.lightning = wx.Lightning(w.get("strikes", []), seed=self.seed)
        self.rain = wx.Rain(self.seed)
        self.ash = wx.Particles("ash", self.seed)
        self.embers = wx.Particles("embers", self.seed + 1)
        self.birds = [wx.Birds(t0, seed=self.seed + i, origin=o, toward=d) for i, (t0, o, d) in enumerate(w.get("birds", []))]
        self.shots = np.array(sorted(w.get("shots", [])), float)

    # ---- the light of the moment
    def mood(self, t):
        """(sky image, land tint, sun, moon, blaze, cast light) crossfaded between moods."""
        cur = [m for m in self.moods if m[0] <= t] or self.moods[:1]
        t0, name = cur[-1]
        k = min(1.0, (t - t0) / self.fade) if len(cur) > 1 else 1.0
        a = MOODS[name]
        sky = self.skies.at(a[0], t)
        vals = [np.array(a[1], np.float32), a[2], a[3], a[4], np.array(a[5], np.float32)]
        if k < 1:
            b = MOODS[cur[-2][1]]
            if b[0] != a[0]:
                sky = sky * k + self.skies.at(b[0], t) * (1 - k)
            vb = [np.array(b[1], np.float32), b[2], b[3], b[4], np.array(b[5], np.float32)]
            vals = [x * k + y * (1 - k) for x, y in zip(vals, vb)]
        return (sky,) + tuple(vals)

    def gunflash(self, t):
        if not len(self.shots):
            return 0.0
        i = int(np.searchsorted(self.shots, t, side="right")) - 1
        return math.exp(-(t - self.shots[i]) / 0.05) if i >= 0 and t - self.shots[i] < 0.3 else 0.0

    # ---- views
    @staticmethod
    def kind(shot):
        names = shot if isinstance(shot, tuple) else (shot,)
        for p in ("land", "walk", "full", "giant"):
            if any(str(n).startswith(p) for n in names):
                return p
        return "clearing"

    def stage(self, song):
        return self

    def giant(self):
        if self._giant is None and self.giant_f is not None:
            self._giant = self.giant_f()
        return self._giant

    def frame(self, t, wide, cue):
        shot = cue["shot"]
        kd = self.kind(shot)
        sky, tint, sun, moon, blaze, _ = self.mood(t)
        f = np.empty((H, W, 3), np.float32)
        if kd == "clearing":
            return self.scn.clearing(f, sky, sun=sun, moon=moon, blaze=blaze * self.flicker(t), tint=tint)
        travel = self.walk_speed * t if kd in ("land", "walk") else 0.0
        if kd == "giant":  # the giant stands behind the trees: sky, far mountains, him, then the trees
            hz = 220
            self.scn.walk(f, travel, sky, tint, horizon=hz, layers=("far",))
            g = self.giant()
            if g is not None:
                g.draw(f, t)
            self.scn.walk(f, travel, None, tint, horizon=hz, layers=("mid",))  # a far ridge of trees in front: his size
            return f
        self.scn.walk(f, travel, sky, tint, horizon=160 if kd == "land" else 0)
        if kd == "land":
            self.scn.ferns(f, self.walk_speed * t, tint)
        return f

    def flicker(self, t):
        return 1 + 0.12 * math.sin(11.3 * t) + 0.08 * math.sin(17.9 * t + 1) + 0.06 * math.sin(5.1 * t + 2)

    def cue(self, t, shot):
        return dict(shot=shot)

    def backdrop(self, frame, t, shot):
        b = key(self.wx_spec.get("blaze", []), t)  # the woods burning behind the cast
        if b > 0:
            kd = self.kind(shot)
            wx.blaze(frame, t, b * self.flicker(t), y=H * {"clearing": 0.62, "giant": 0.8}.get(kd, 0.72))
        self.keep_bg(frame)

    def fore(self, frame, t, shot):
        w = self.wx_spec
        kd = self.kind(shot)
        _, tint, sun, moon, blaze, cast = self.mood(t)
        fl = self.lightning.flash(t)
        g = self.gunflash(t)
        self.light_cast(frame, cast * (1 + 0.8 * fl) + np.array([0.3, 0.6, 0.9], np.float32) * g)
        if kd in ("walk", "full"):
            self.scn.ferns(frame, self.walk_speed * t if kd == "walk" else 0.0, tint)
        sm = key(w.get("smoke", []), t)
        if sm > 0:
            wx.smoke(frame, t, sm)
        for bd in self.birds:
            if kd in ("land", "full", "walk", "giant") or abs(t - bd.t0) < 3:
                bd.draw(frame, t)
        r = key(w.get("rain", []), t)
        if r > 0:
            self.rain.draw(frame, t, r)
        a = key(w.get("ash", []), t)
        if a > 0:
            self.ash.draw(frame, t, a)
        e = key(w.get("embers", []), t)
        if e > 0:
            self.embers.draw(frame, t, e)
        if fl > 0:
            self.lightning.draw(frame, t)
        if g > 0:
            frame += 0.25 * g * np.array([0.6, 0.85, 1.0], np.float32)
            np.minimum(frame, 1.0, out=frame)

    def closeup_light(self, t):
        return "sunset"
