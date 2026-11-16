"""Weather and wild effects drawn in 2D over any frame, all stateless (a function of t), so any
frame renders on its own:

  Lightning(strikes, seed).draw(frame, t)      # strikes: [(t, x)]: a forked bolt + a flash of the sky
  Lightning.flash(t)                           # 0..1, to light the cast or the plate too
  Rain(seed).draw(frame, t, amount)            # slanted streaks, a little mist
  Particles(kind, seed).draw(frame, t, amount) # "ash" drifting down, "embers" rising
  Birds(t0, seed, n, origin).draw(frame, t)    # a flock bursting up out of the trees at t0
  smoke(frame, t, amount, color)               # soft smoke rising over the picture
  blaze(frame, t, amount, y)                   # the woods burning: a flickering glow under a treeline

Colors are BGR floats in display space (production.Production's frames).
"""
import math

import cv2
import numpy as np

from production import H, W


def _bolt(rng, x0, y0, y1, depth=0):
    """A jagged path from (x0, y0) down to y1, with forks: list of polylines with widths."""
    pts, x, y = [(x0, y0)], x0, y0
    step = (y1 - y0) / rng.integers(14, 22)
    while y < y1:
        y += step * rng.uniform(0.6, 1.3)
        x += rng.normal(0, abs(step) * 0.55)
        pts.append((x, min(y, y1)))
    out = [(np.array(pts, np.float32), 1.0 if depth == 0 else 0.45)]
    if depth < 2:
        for _ in range(int(rng.integers(1, 4 - depth))):
            i = int(rng.integers(2, max(3, len(pts) - 3)))
            fx, fy = pts[i]
            out += [(p, w * 0.6) for p, w in _bolt(rng, fx, fy, fy + (y1 - fy) * rng.uniform(0.3, 0.7), depth + 1)]
    return out


class Lightning:
    def __init__(self, strikes, seed=3, ground=None):
        """strikes: [(t, x)] (x in px, None: random); ground: y where a bolt ends (default: 80% down)."""
        rng = np.random.default_rng(seed)
        self.strikes = []
        for t, x in strikes:
            x = float(rng.uniform(0.15, 0.85) * W) if x is None else float(x)
            self.strikes.append((float(t), _bolt(rng, x + rng.normal(0, 80), -20, ground or H * 0.8)))
        self.ts = np.array([s[0] for s in self.strikes], float)

    def flash(self, t):
        """How bright the sky is from lightning now (0..1): a double flicker, then a tail."""
        k = 0.0
        for t0 in self.ts[(self.ts <= t) & (self.ts > t - 0.6)]:
            u = t - t0
            k = max(k, math.exp(-u / 0.05) + 0.6 * math.exp(-max(0.0, u - 0.09) / 0.07) * (u > 0.09))
        return min(1.0, k)

    def draw(self, frame, t, sky_mask=None):
        k = self.flash(t)
        if k <= 0.01:
            return
        if sky_mask is not None:
            frame += (0.55 * k) * sky_mask * np.array([1.0, 0.95, 0.9], np.float32)
        frame += 0.12 * k
        for t0, paths in self.strikes:
            u = t - t0
            if not (0 <= u < 0.25):
                continue
            a = math.exp(-u / 0.06) + 0.7 * math.exp(-max(0.0, u - 0.09) / 0.05) * (u > 0.09)
            glow = np.zeros((H // 2, W // 2), np.float32)
            core = np.zeros((H, W), np.float32)
            for p, w in paths:
                q = p.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(core, [q], False, 1.0, max(1, int(4 * w)), cv2.LINE_AA)
                cv2.polylines(glow, [(q // 2)], False, 1.0, max(1, int(6 * w)), cv2.LINE_AA)
            glow = cv2.resize(cv2.GaussianBlur(glow, (0, 0), 12), (W, H))
            frame += (min(1.0, a) * (core[..., None] * np.array([1.0, 0.95, 0.92], np.float32) + 1.2 * glow[..., None] * np.array([1.0, 0.8, 0.7], np.float32)))
        np.minimum(frame, 1.0, out=frame)


class Rain:
    def __init__(self, seed=4, n=900, slant=0.18):
        rng = np.random.default_rng(seed)
        self.x, self.y = rng.uniform(0, W * 1.2, n), rng.uniform(0, H, n)
        self.v = rng.uniform(1800, 2600, n)
        self.L = rng.uniform(20, 46, n)
        self.b = rng.uniform(0.15, 0.4, n)
        self.slant = slant

    def draw(self, frame, t, amount=1.0):
        if amount <= 0:
            return
        n = int(len(self.x) * min(1.0, amount))
        y = (self.y[:n] + self.v[:n] * t) % (H + 60) - 30
        x = (self.x[:n] - self.slant * self.v[:n] * t) % (W * 1.2) - W * 0.1
        lay = np.zeros((H, W), np.float32)
        for xi, yi, L, b in zip(x, y, self.L[:n], self.b[:n]):
            cv2.line(lay, (int(xi), int(yi)), (int(xi - self.slant * L), int(yi + L)), float(b), 1, cv2.LINE_AA)
        a = (lay * amount)[..., None]
        frame[:] = frame * (1 - 0.6 * a) + a * np.array([0.85, 0.85, 0.82], np.float32)
        frame *= 1 - 0.12 * amount  # wet, dim
        frame += 0.05 * amount * np.array([0.5, 0.5, 0.48], np.float32)  # mist


class Particles:
    def __init__(self, kind="ash", seed=6, n=260):
        rng = np.random.default_rng(seed)
        self.kind = kind
        self.x, self.y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        self.r = rng.uniform(1.2, 3.8, n) * (1.0 if kind == "ash" else 0.8)
        self.v = rng.uniform(25, 90, n) * (1 if kind == "ash" else -2.2)
        self.ph = rng.uniform(0, 6.28, n)
        self.b = rng.uniform(0.4, 1.0, n)

    def draw(self, frame, t, amount=1.0):
        if amount <= 0:
            return
        n = int(len(self.x) * min(1.0, amount))
        y = (self.y[:n] + self.v[:n] * t) % (H + 40) - 20
        x = (self.x[:n] + 30 * np.sin(t * 0.7 + self.ph[:n]) + 12 * t) % W
        lay = np.zeros((H, W), np.float32)
        for xi, yi, r, b in zip(x, y, self.r[:n], self.b[:n]):
            cv2.circle(lay, (int(xi), int(yi)), max(1, int(r)), float(b), -1, cv2.LINE_AA)
        if self.kind == "ash":
            a = (cv2.GaussianBlur(lay, (0, 0), 0.8) * 0.7 * amount)[..., None]
            frame[:] = frame * (1 - a) + a * np.array([0.72, 0.74, 0.76], np.float32)
        else:
            fl = 0.6 + 0.4 * np.sin(t * 9.0)
            g = cv2.GaussianBlur(lay, (0, 0), 3.0) * 1.5 + lay
            frame += (g * amount * fl)[..., None] * np.array([0.15, 0.55, 1.0], np.float32)
            np.minimum(frame, 1.0, out=frame)


class Birds:
    def __init__(self, t0, seed=7, n=24, origin=(W * 0.5, H * 0.62), spread=420, dur=4.0, toward=+1):
        """A flock bursting up out of the trees at t0 and flying off to one side (toward=+1: right)."""
        rng = np.random.default_rng(seed)
        self.t0, self.dur, self.n = t0, dur, n
        self.o = np.array(origin, float) + np.c_[rng.normal(0, spread * 0.4, n), rng.normal(0, 30, n)]
        self.vel = np.c_[toward * rng.uniform(150, 420, n), -rng.uniform(260, 520, n)]
        self.delay = rng.uniform(0, 0.35, n)
        self.size = rng.uniform(9, 20, n)
        self.flap = rng.uniform(9, 14, n)
        self.ph = rng.uniform(0, 6.28, n)

    def draw(self, frame, t, color=(0.06, 0.05, 0.05)):
        u = t - self.t0
        if not (0 <= u < self.dur):
            return
        lay = np.zeros((H, W), np.float32)
        for i in range(self.n):
            v = u - self.delay[i]
            if v < 0:
                continue
            p = self.o[i] + self.vel[i] * v + np.array([0, -60 * v * v])  # climbing
            s = self.size[i] * (1 + 0.25 * v)
            w = math.sin(self.flap[i] * v * 2 * math.pi / 3 + self.ph[i])
            tip = s * 0.55 * w
            pts = np.array([[p[0] - s, p[1] - tip], [p[0] - s * 0.45, p[1] - tip * 0.4 - s * 0.12], [p[0], p[1] + s * 0.12],
                            [p[0] + s * 0.45, p[1] - tip * 0.4 - s * 0.12], [p[0] + s, p[1] - tip]], np.int32)
            cv2.polylines(lay, [pts], False, 1.0, max(2, int(s * 0.22)), cv2.LINE_AA)
        a = lay[..., None] * min(1.0, (self.dur - u) / 0.6)
        frame[:] = frame * (1 - a) + a * np.array(color, np.float32)


_smoke_cache = {}


def smoke(frame, t, amount=1.0, color=(0.10, 0.10, 0.11), speed=40.0, seed=9):
    """Soft smoke rising over the picture (thicker low down)."""
    if amount <= 0:
        return
    if seed not in _smoke_cache:
        rng = np.random.default_rng(seed)
        n = np.zeros((H // 4, W // 4), np.float32)
        for o, s in ((1.0, 0.5), (0.5, 1.2), (0.25, 3.0)):
            g = rng.normal(0, 1, (int(H // 4 * s / 8) + 2, int(W // 4 * s / 8) + 2)).astype(np.float32)
            n += o * cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_CUBIC)
        n = (n - n.min()) / (n.max() - n.min())
        _smoke_cache[seed] = np.concatenate([n, n[::-1]], 0)
    n = _smoke_cache[seed]
    off = int(t * speed / 4) % n.shape[0]
    sm = np.roll(n, -off, axis=0)[:H // 4]
    yy = np.linspace(0, 1, H // 4, dtype=np.float32)[:, None]
    a = np.clip((sm - 0.45) * 2.5, 0, 1) * (0.35 + 0.65 * yy) * amount
    a = cv2.resize(a, (W, H))[..., None]
    frame[:] = frame * (1 - a) + a * np.array(color, np.float32)


def blaze(frame, t, amount=1.0, y=H * 0.75, seed=11):
    """The woods on fire somewhere behind: a flickering orange glow rising from y (added light)."""
    if amount <= 0:
        return
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    yy = np.arange(H, dtype=np.float32)[:, None]
    fl = 0.75 + 0.25 * np.sin(xx * 23 + t * 7.3) * np.sin(xx * 7 - t * 3.1 + 1) + 0.1 * math.sin(t * 13)
    up = np.where(yy < y, np.clip(1 - (y - yy) / (H * 0.55), 0, 1) ** 2, np.exp(-((yy - y) / (H * 0.12)) ** 2))
    g = (up * fl * amount)[..., None] * np.array([0.05, 0.28, 0.75], np.float32)
    frame += g
    np.minimum(frame, 1.0, out=frame)
