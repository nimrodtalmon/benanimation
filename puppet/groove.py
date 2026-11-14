"""Groove: how a player's body moves to the music, from the score alone (stateless, so any frame
can be rendered on its own and clips render in parallel).

  g = Groove(song, seed=3, amount=1.0)
  m = g.at(t)   # dict(rot=deg, dx=px, dy=px, nod=px) in "drawing at scale 1" units; callers scale

- energy: how hard the band plays now (drum-hit density, smoothed; a bass-only intro is low).
- sway: a slow rock over two bars, plus a lean that changes each 4-bar phrase (seeded, so the
  players do not move in unison).
- bounce: a small knee dip after each beat.
- head: nods on beats when the band is loud (stronger on the bar's first beat), with a little
  overshoot so the head settles instead of stopping dead.
- breaks (score "holes", stop-time): everyone hits the break together (a quick lean back) and
  holds still until the band comes back in.
"""
import math

import numpy as np

FPS = 30


def _impulse(u, tau):
    """A soft kick: rises fast, decays (u >= 0), peak 1 at u = tau."""
    u = np.maximum(u, 0)
    return (u / tau) * np.exp(1 - u / tau)


def _settle(u, tau, freq):
    """A nod that overshoots a little and settles."""
    u = np.maximum(u, 0)
    return np.exp(-u / tau) * np.sin(2 * math.pi * freq * u) * (u < 6 * tau)


class Groove:
    def __init__(self, song, seed=0, amount=1.0, sway=2.2, bounce=6.0, nod=9.0, step=10.0, lean=2.0):
        self.song, self.amount = song, amount
        self.k = dict(sway=sway, bounce=bounce, nod=nod, step=step, lean=lean)
        d = song.data
        self.beats = np.array(d.get("beats") or np.arange(0, song.duration, song.spb))
        self.holes = [tuple(h) for h in d.get("holes", [])]
        n = int(song.duration * FPS) + 2
        # energy: drum hits per second, smoothed over ~1.5 s, normalized; the bass alone gives a little
        hits = np.zeros(n)
        for e in d.get("drums", []):
            i = int(e["t"] * FPS)
            if 0 <= i < n:
                hits[i] += 1
        bass = np.zeros(n)
        for e in d.get("bass", []):
            i = int(e["t"] * FPS)
            if 0 <= i < n:
                bass[i] += 1
        ker = np.hanning(int(1.5 * FPS))
        ker /= ker.sum()
        dr = np.convolve(hits, ker, mode="same") * FPS
        bs = np.convolve(bass, ker, mode="same") * FPS
        ref = np.percentile(dr[dr > 0], 80) if (dr > 0).any() else 1
        self.energy = np.clip(0.8 * dr / ref + 0.35 * np.clip(bs / 3, 0, 1), 0, 1.1)
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * math.pi, 3)
        n_phr = int(song.duration / (4 * song.bar)) + 3
        self.leans = rng.uniform(-1, 1, n_phr)
        self.leans[0] = 0.0

    def e(self, t):
        i = int(np.clip(t * FPS, 0, len(self.energy) - 1))
        return float(self.energy[i])

    def in_hole(self, t):
        for a, b in self.holes:
            if a - 0.05 <= t < b:
                return a, b
        return None

    def at(self, t):
        A, k, song = self.amount, self.k, self.song
        e = self.e(t)
        bar = song.bar
        # slow rock over two bars, and a lean held for each 4-bar phrase (eased between phrases)
        rot = k["sway"] * e * math.sin(2 * math.pi * t / (2 * bar) + self.phase[0])
        ph = t / (4 * bar)
        i = int(ph)
        u = ph - i
        w = 0.5 - 0.5 * math.cos(math.pi * min(1.0, u / 0.15))  # the first 15% of a phrase: move to the new lean
        lean = self.leans[i] * w + self.leans[max(0, i - 1)] * (1 - w)
        rot += k["lean"] * lean * (0.4 + 0.6 * e)
        dx = k["step"] * e * math.sin(2 * math.pi * t / (2 * bar) + self.phase[1]) + 0.5 * k["step"] * lean
        # beats: a knee dip after each beat, head nods (bar starts stronger)
        j = np.searchsorted(self.beats, t, side="right")
        recent = self.beats[max(0, j - 4):j]
        dy, nod = 0.0, 0.0
        for b in recent:
            idx = int(np.searchsorted(self.beats, b))
            down = 1.0 if idx % 4 == 0 else 0.6
            dy += float(_impulse(t - b, 0.09)) * (0.7 + 0.3 * down)
            nod += float(_settle(t - b, 0.16, 1.6)) * down
        dy *= k["bounce"] * e
        nod *= k["nod"] * max(0.0, e - 0.25) ** 1.2
        # stop-time: hit the break (lean back), freeze, and come back in (smooth in and out)
        hf, kick = 0.0, 0.0
        for a_, b_ in self.holes:
            if a_ - 0.1 <= t < b_ + 0.3:
                hf = max(hf, min(1.0, max(0.0, (t - a_ + 0.02) / 0.06)) * min(1.0, max(0.0, (b_ + 0.1 - t) / 0.12)))
            if a_ <= t < a_ + 1.0:
                kick += float(_impulse(t - a_, 0.07))
        rot = rot * (1 - 0.7 * hf) - 1.8 * kick
        dy = dy * (1 - hf) - 6 * kick
        nod = nod * (1 - hf) - 6 * kick
        dx = dx * (1 - 0.5 * hf)
        return dict(rot=A * rot, dx=A * dx, dy=A * dy, nod=A * nod, energy=e, hole=hf > 0.5)
