"""Outdoor scenery, all drawn in code (flat marker colors, to sit with Ben's drawings):

  Sky(seed).frame(dusk, t, travel, clearing)  sky (sunset -> night: sun, stars, moon), far hills,
                                              a pine ridge, near pines; travel (px) scrolls the
                                              layers at their own speeds (parallax walking);
                                              clearing=True: the trees step back, a ground
  Sky.ferns(frame, t, travel, dusk)           ferns and grass passing in front (walking shots)
  firelight(frame, pos, heat, dusk, ...)      ambient (dusk -> moonlight) + warm flickering light
                                              falling off from the fire
  Fire(seed).draw(frame, t, pos, scale, heat) a campfire: stones, logs, embers, cartoon flames
                                              (banded colors, licking tongues), glow
  Sparks(duration, heat_fn, bursts).draw(...) sparks: a steady stream with the heat, bursts on
                                              given times; stateless (any frame alone)

Frames are float32 BGR in [0, 1], H x W (production.H, W).
"""
import math

import cv2
import numpy as np

from production import H, W

LW = 4096  # the scrolling layers wrap around this width


def _lerp(a, b, k):
    return np.asarray(a, np.float32) * (1 - k) + np.asarray(b, np.float32) * k


def _ridge(rng, base, amp, n=7):
    x = np.arange(LW, dtype=np.float32)
    y = np.full(LW, base, np.float32)
    for k in range(1, n + 1):
        y += amp / k ** 0.9 * rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * k * x / LW * rng.integers(1, 4) + rng.uniform(0, 6.3))
    return y


def _pine(m, x, base, h, w, rng):
    tiers = rng.integers(4, 7)
    cv2.rectangle(m, (int(x - w * 0.06), int(base - h * 0.15)), (int(x + w * 0.06), int(base)), 255, -1)
    for i in range(tiers):
        k0, k1 = i / tiers, (i + 1.6) / tiers
        yb, yt = base - h * (0.12 + 0.8 * k0), base - h * min(1.0, 0.12 + 0.8 * k1 + 0.08)
        hw = w * 0.5 * (1 - 0.75 * k0) * rng.uniform(0.85, 1.1)
        jag = rng.uniform(-0.08, 0.08) * w
        pts = np.array([[x - hw, yb], [x - hw * 0.3, yb - h * 0.03], [x + jag * 0.3, yt], [x + hw * 0.3, yb - h * 0.04], [x + hw, yb]], np.int32)
        cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)


def _wrap_draw(m, fn, x, *a):
    for dx in (-LW, 0, LW):
        if -400 < x + dx < LW + 400:
            fn(m, x + dx, *a)


class Sky:
    def __init__(self, seed=5):
        rng = np.random.default_rng(seed)
        self.rng_seed = seed
        xs = np.arange(LW)
        yy = np.arange(H, dtype=np.float32)[:, None]
        # far hills: a soft ridge
        self.far = (yy >= _ridge(rng, H * 0.56, 55)[None, :]).astype(np.uint8) * 255
        # the pine ridge: hills topped with small pines
        r = _ridge(rng, H * 0.66, 30)
        mid = (yy >= r[None, :]).astype(np.uint8) * 255
        for x in np.cumsum(rng.uniform(14, 34, 400)):
            if x >= LW:
                break
            _wrap_draw(mid, lambda m, x, b, h, w: _pine(m, x, b, h, w, rng), x, r[int(x) % LW] + 6, rng.uniform(40, 90), rng.uniform(26, 40))
        self.mid = mid
        # near pines (walking: they pass by), and the edge of the clearing (smaller, further)
        near = np.zeros((H, LW), np.uint8)
        for x in np.cumsum(rng.uniform(140, 420, 60)):
            if x >= LW:
                break
            h = rng.uniform(520, 900)
            _wrap_draw(near, lambda m, x, b, h, w: _pine(m, x, b, h, w, rng), x, H * 0.9, h, h * rng.uniform(0.38, 0.5))
        cv2.rectangle(near, (0, int(H * 0.86)), (LW, H), 255, -1)
        self.near = near
        edge = np.zeros((H, LW), np.uint8)
        for x in np.cumsum(rng.uniform(40, 110, 120)):
            if x >= LW:
                break
            h = rng.uniform(170, 330)
            _wrap_draw(edge, lambda m, x, b, h, w: _pine(m, x, b, h, w, rng), x, H * 0.73 + rng.uniform(-8, 8), h, h * rng.uniform(0.4, 0.55))
        self.edge = edge
        # the ground of the clearing: grass tufts on its rim
        g = (yy >= _ridge(rng, H * 0.78, 6)[None, :]).astype(np.uint8) * 255
        for x in np.cumsum(rng.uniform(5, 16, 900)):
            if x >= LW:
                break
            y0 = int(H * 0.78) + 4
            for k in range(3):
                cv2.line(g, (int(x), y0), (int(x + rng.uniform(-9, 9)), int(y0 - rng.uniform(8, 26))), 255, 2, cv2.LINE_AA)
        self.ground = g
        # stars
        n = 650
        self.stars = (rng.uniform(0, W, n), rng.uniform(0, H * 0.62, n) ** 1.0, rng.uniform(0.3, 1.0, n) ** 2, rng.uniform(0, 6.3, n),
                      rng.uniform(1.0, 3.5, n))
        # ferns and grass in front (walking)
        f = np.zeros((H, LW), np.uint8)
        for x in np.cumsum(rng.uniform(35, 110, 140)):
            if x >= LW:
                break
            kind = rng.uniform()
            if kind < 0.5:  # a fern: fronds arching out of a clump
                for k in range(rng.integers(5, 9)):
                    a = math.radians(rng.uniform(-75, 75))
                    L = rng.uniform(200, 360)
                    pts = []
                    for s in np.linspace(0, 1, 14):
                        bend = a + math.copysign(1, a or 1) * 0.9 * s * s
                        pts.append((x + math.sin(bend) * L * s, H + 30 - math.cos(bend) * L * s * 0.9))
                    for i, (px, py) in enumerate(pts[1:], 1):
                        lw = 26 * (1 - i / 14) + 4
                        cv2.line(f, tuple(map(int, pts[i - 1])), (int(px), int(py)), 255, 6, cv2.LINE_AA)
                        nx, ny = py - pts[i - 1][1], -(px - pts[i - 1][0])
                        nn = math.hypot(nx, ny) or 1
                        for sg in (-1, 1):
                            cv2.line(f, (int(px), int(py)), (int(px + sg * nx / nn * lw), int(py + sg * ny / nn * lw - 6)), 255, 4, cv2.LINE_AA)
            else:  # a bush and some grass
                _wrap_draw(f, lambda m, x, rx, ry: cv2.ellipse(m, (int(x), H + 20), (int(rx), int(ry)), 0, 0, 360, 255, -1, cv2.LINE_AA),
                           x, rng.uniform(110, 220), rng.uniform(130, 240))
                for k in range(25):
                    x0 = x + rng.uniform(-150, 150)
                    cv2.line(f, (int(x0) % LW, H), (int(x0 + rng.uniform(-30, 30)) % LW, int(H - rng.uniform(60, 190))), 255, 3, cv2.LINE_AA)
        self.ferns_m = f
        self.yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]

    def _layer(self, frame, m, off, col):
        idx = (np.arange(W) + int(off)) % LW
        a = (m[:, idx].astype(np.float32) / 255.0)[..., None]
        frame *= 1 - a
        frame += a * np.asarray(col, np.float32)

    def frame(self, d, t, travel=0.0, clearing=False):
        yy = self.yy
        hz = 0.62
        top = _lerp((0.42, 0.22, 0.24), (0.10, 0.04, 0.025), d)
        mid = _lerp((0.50, 0.42, 0.80), (0.20, 0.09, 0.05), d)
        low = _lerp((0.30, 0.62, 1.00), (0.30, 0.16, 0.09), d)
        k = np.clip(yy / hz, 0, 1)
        f = np.where(k < 0.6, top * (1 - k / 0.6) + mid * (k / 0.6), mid * (1 - (k - 0.6) / 0.4) + low * ((k - 0.6) / 0.4)).astype(np.float32)
        f = np.ascontiguousarray(np.broadcast_to(f, (H, W, 3)))
        # the sun, sinking
        if d < 0.75:
            sx, sy = W * 0.3 - 0.02 * travel, H * (0.5 + 0.25 * d)
            sx = (sx % (W * 1.4)) - W * 0.2
            glow = np.zeros((H // 4, W // 4), np.float32)
            cv2.circle(glow, (int(sx / 4), int(sy / 4)), 24, 1.0, -1, cv2.LINE_AA)
            g = cv2.resize(cv2.GaussianBlur(glow, (0, 0), 30), (W, H))[..., None]
            f += (1 - d / 0.75) * (0.9 * g * np.array([0.2, 0.5, 0.7], np.float32))
            cv2.circle(f, (int(sx), int(sy)), 95, tuple(float(c) for c in _lerp((0.55, 0.88, 1.0), (0.3, 0.45, 0.9), d / 0.75)), -1, cv2.LINE_AA)
        # stars and the moon
        ks = float(np.clip((d - 0.45) / 0.4, 0, 1))
        if ks > 0:
            x, y, b, ph, sp = self.stars
            tw = b * (0.75 + 0.25 * np.sin(t * sp + ph)) * ks
            xi = ((x - 0.01 * travel) % W).astype(int)
            yi = y.astype(int)
            for i in np.nonzero(tw > 0.06)[0]:
                v = float(tw[i])
                cv2.circle(f, (xi[i], yi[i]), 2 if b[i] > 0.6 else 1, (0.8 * v + 0.1, 0.85 * v + 0.08, 0.9 * v + 0.05), -1, cv2.LINE_AA)
            mx, my = int(W * 0.8), int(H * 0.15)
            gl = np.zeros((H // 4, W // 4), np.float32)
            cv2.circle(gl, (mx // 4, my // 4), 14, 1.0, -1)
            f += ks * 0.35 * cv2.resize(cv2.GaussianBlur(gl, (0, 0), 18), (W, H))[..., None] * np.array([0.9, 0.85, 0.7], np.float32)
            moon = np.zeros((160, 160), np.float32)
            cv2.circle(moon, (80, 80), 56, 1.0, -1, cv2.LINE_AA)
            cv2.circle(moon, (104, 68), 50, 0.0, -1, cv2.LINE_AA)  # a crescent
            sub = f[my - 80:my + 80, mx - 80:mx + 80]
            a = (ks * moon)[..., None]
            sub *= 1 - a
            sub += a * np.array([0.85, 0.94, 0.97], np.float32)
        # the layers, far to near
        self._layer(f, self.far, 0.04 * travel, _lerp((0.55, 0.38, 0.48), (0.17, 0.09, 0.07), d))
        self._layer(f, self.mid, 0.16 * travel, _lerp((0.30, 0.20, 0.25), (0.09, 0.05, 0.04), d))
        if clearing:
            self._layer(f, self.edge, 0.0, _lerp((0.14, 0.10, 0.12), (0.05, 0.03, 0.025), d))
            self._layer(f, self.ground, 0.0, _lerp((0.10, 0.14, 0.16), (0.05, 0.06, 0.07), d))
        else:
            self._layer(f, self.near, 0.55 * travel, _lerp((0.10, 0.08, 0.10), (0.035, 0.025, 0.02), d))
        return f

    def ferns(self, frame, t, travel, d):
        self._layer(frame, self.ferns_m, 1.5 * travel, _lerp((0.05, 0.07, 0.06), (0.02, 0.025, 0.02), d))


def _flicker(t, seed=0):
    return 1 + 0.07 * math.sin(11.3 * t + seed) + 0.05 * math.sin(17.9 * t + 2 * seed) + 0.04 * math.sin(29.1 * t + 3)


def firelight(frame, pos, heat, dusk=1.0, glow=0.0, sun=0.0, t=0.0):
    """Ambient (warm at sunset, blue moonlight at night) times the picture, plus warm light from the
    fire at pos (it can be off-screen), falling off with distance, flickering."""
    h4, w4 = H // 4, W // 4
    if not hasattr(firelight, "g"):
        firelight.g = np.mgrid[0:h4, 0:w4].astype(np.float32) * 4
    yy, xx = firelight.g
    amb = _lerp((0.95, 0.92, 0.95), (0.62, 0.55, 0.50), dusk)
    r = np.hypot(xx - pos[0], (yy - pos[1]) * 1.15)
    fall = np.exp(-r / 700.0)
    k = heat * _flicker(t, 1)
    m = amb[None, None, :] + (k * 0.85 + glow) * fall[..., None] * np.array([0.30, 0.62, 1.05], np.float32)
    if sun > 0:
        m = m + sun * 0.12 * np.clip(1 - xx / W, 0, 1)[..., None] * np.array([0.0, 0.25, 0.45], np.float32)
    frame *= cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return frame


class Fire:
    def __init__(self, seed=5):
        rng = np.random.default_rng(seed)
        self.tongues = [(c, rng.uniform(0.55, 1.0), rng.uniform(0.28, 0.42), rng.uniform(0, 6.3), rng.uniform(0.8, 1.3))
                        for c in (-0.42, -0.2, 0.0, 0.22, 0.44, -0.05, 0.1)]
        self.stones = [(math.radians(a + rng.uniform(-8, 8)), rng.uniform(0.85, 1.15), rng.uniform(0.6, 0.85)) for a in range(0, 360, 30)]

    def flames(self, t, heat, w, h):
        """The flame intensity on a w x h canvas (base at the bottom middle)."""
        v = np.linspace(1, 0, h, dtype=np.float32)[:, None]
        u = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        I = np.zeros((h, w), np.float32)
        for c, hh, ww, ph, sp in self.tongues:
            hi = hh * (0.75 + 0.25 * math.sin(t * 5.3 * sp + ph)) * (0.12 + 0.62 * min(1.6, heat))
            vt = v / max(hi, 1e-3)
            dsp = 0.16 * vt * (np.sin(6.5 * vt - 9 * sp * t + ph) + 0.5 * np.sin(13 * vt - 15 * t + 2 * ph))
            wv = ww * np.clip(1 - vt, 0, 1) ** 0.75 * (0.55 + 0.45 * np.clip(vt * 6, 0, 1))
            I = np.maximum(I, np.clip(1 - np.abs(u - c * (1 - 0.5 * vt) - dsp) / np.maximum(wv, 1e-3), 0, 1) * (vt < 1))
        return I

    def draw(self, frame, t, pos, sc, heat):
        x, y = pos
        heat = float(heat)
        # stones behind, embers, logs, flames, stones in front
        rx, ry = 230 * sc, 46 * sc

        def stones(front):
            for a, s, g in self.stones:
                sy = math.sin(a)
                if (sy > 0) != front:
                    continue
                cx, cy = x + math.cos(a) * rx, y - 18 * sc + sy * ry
                col = (0.30 * g, 0.31 * g, 0.33 * g)
                cv2.ellipse(frame, (int(cx), int(cy)), (int(34 * sc * s), int(22 * sc * s)), 0, 0, 360, col, -1, cv2.LINE_AA)
                cv2.ellipse(frame, (int(cx), int(cy)), (int(34 * sc * s), int(22 * sc * s)), 0, 0, 360, (0.04, 0.04, 0.05), max(1, int(3 * sc)), cv2.LINE_AA)
                cv2.ellipse(frame, (int(cx), int(cy - 6 * sc)), (int(20 * sc * s), int(8 * sc * s)), 0, 180, 360,
                            (0.25 * heat + 0.3 * g, 0.45 * heat + 0.3 * g, 0.7 * heat + 0.3 * g), max(1, int(3 * sc)), cv2.LINE_AA)
        stones(False)
        cv2.ellipse(frame, (int(x), int(y - 16 * sc)), (int(150 * sc), int(30 * sc)), 0, 0, 360,
                    (0.05 + 0.1 * heat, 0.12 + 0.3 * heat, 0.35 + 0.6 * heat), -1, cv2.LINE_AA)
        for ang, dx in ((-14, -10), (12, 12), (-3, 0)):
            L, th = 330 * sc, 40 * sc
            a = math.radians(ang)
            c, s = math.cos(a), math.sin(a)
            cxl, cyl = x + dx * sc, y - 28 * sc
            pts = np.array([[cxl - c * L / 2 - s * th / 2, cyl - s * L / 2 + c * th / 2], [cxl + c * L / 2 - s * th / 2, cyl + s * L / 2 + c * th / 2],
                            [cxl + c * L / 2 + s * th / 2, cyl + s * L / 2 - c * th / 2], [cxl - c * L / 2 + s * th / 2, cyl - s * L / 2 - c * th / 2]], np.int32)
            cv2.fillPoly(frame, [pts], (0.09, 0.15, 0.24), cv2.LINE_AA)
            cv2.polylines(frame, [pts], True, (0.02, 0.03, 0.05), max(1, int(4 * sc)), cv2.LINE_AA)
            cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (0.15 + 0.2 * heat, 0.3 + 0.4 * heat, 0.5 + 0.5 * heat), max(1, int(4 * sc)), cv2.LINE_AA)
            for e in (0, 1):  # the cut ends, rings
                ex, ey = (pts[0] + pts[3]) / 2 if e == 0 else (pts[1] + pts[2]) / 2
                cv2.ellipse(frame, (int(ex), int(ey)), (int(10 * sc), int(th / 2)), ang, 0, 360, (0.25, 0.42, 0.55), -1, cv2.LINE_AA)
                cv2.ellipse(frame, (int(ex), int(ey)), (int(10 * sc), int(th / 2)), ang, 0, 360, (0.02, 0.03, 0.05), max(1, int(3 * sc)), cv2.LINE_AA)
        # the flames
        fw, fh = int(340 * sc), int(520 * sc)
        I = self.flames(t, heat, fw // 2, fh // 2)
        I = cv2.resize(I, (fw, fh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = int(x - fw / 2), int(y - 30 * sc - fh)
        bands = [(0.06, (0.06, 0.12, 0.80)), (0.20, (0.08, 0.42, 1.0)), (0.45, (0.25, 0.82, 1.0)), (0.72, (0.70, 0.96, 1.0))]
        col = np.zeros((fh, fw, 3), np.float32)
        al = np.zeros((fh, fw), np.float32)
        for th_, c in bands:
            m = np.clip((I - th_) / 0.03, 0, 1)
            col = col * (1 - m[..., None]) + m[..., None] * np.array(c, np.float32)
            al = np.maximum(al, m)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + fw), min(H, y0 + fh)
        if xb > xa and yb > ya:
            sub = frame[ya:yb, xa:xb]
            a_ = al[ya - y0:yb - y0, xa - x0:xb - x0, None]
            sub *= 1 - a_
            sub += a_ * col[ya - y0:yb - y0, xa - x0:xb - x0]
        # the glow around it
        g = np.zeros((H // 8, W // 8), np.float32)
        cv2.ellipse(g, (int(x / 8), int((y - 150 * sc) / 8)), (int(30 * sc), int(36 * sc)), 0, 0, 360, 1.0, -1)
        g = cv2.resize(cv2.GaussianBlur(g, (0, 0), 14 * sc + 2), (W, H))[..., None]
        frame += (0.55 * min(1.6, heat) * _flicker(t)) * g * np.array([0.15, 0.45, 1.0], np.float32)
        stones(True)


class Sparks:
    def __init__(self, duration, heat_fn, bursts=(), rate=14.0, seed=5):
        rng = np.random.default_rng(seed + 11)
        births = []
        for k in range(int(duration * 30)):
            tk = k / 30.0
            births += list(tk + rng.uniform(0, 1 / 30, rng.poisson(rate * max(0.0, heat_fn(tk) - 0.2) / 30)))
        for b in bursts:
            births += list(b + rng.uniform(0, 0.12, 14))
        n = len(births)
        self.p = dict(t0=np.array(births), u0=rng.normal(0, 70, n), vx=rng.normal(0, 45, n), vy=rng.uniform(220, 520, n),
                      life=rng.uniform(0.7, 2.2, n), wob=rng.uniform(8, 40, n), fq=rng.uniform(2, 6, n), ph=rng.uniform(0, 6.3, n),
                      r=rng.uniform(1.5, 3.8, n))

    def at(self, a, i, x, y, sc):
        p = self.p
        return (x + sc * (p["u0"][i] + p["vx"][i] * a + p["wob"][i] * np.sin(p["fq"][i] * a + p["ph"][i])),
                y - sc * (120 + p["vy"][i] * a - 70 * a * a))

    def draw(self, frame, t, pos, sc):
        p = self.p
        a = t - p["t0"]
        i = np.nonzero((a >= 0) & (a < p["life"]))[0]
        if not len(i):
            return
        a = a[i]
        x, y = self.at(a, i, pos[0], pos[1], sc)
        x1, y1 = self.at(np.maximum(0, a - 0.05), i, pos[0], pos[1], sc)
        lay = np.zeros((H // 2, W // 2, 3), np.float32)
        k = (1 - a / p["life"][i]) ** 1.2
        for j in range(len(i)):
            if not (-20 < x[j] < W + 20 and -20 < y[j] < H + 20):
                continue
            age = 1 - k[j]
            c = _lerp((0.55, 0.95, 1.0), (0.05, 0.25, 1.0), min(1.0, age * 1.4)) * k[j]
            cv2.line(lay, (int(x1[j] / 2), int(y1[j] / 2)), (int(x[j] / 2), int(y[j] / 2)), tuple(float(v) for v in c),
                     max(1, int(p["r"][i[j]] * sc / 2)), cv2.LINE_AA)
        lay = cv2.resize(lay + 0.6 * cv2.GaussianBlur(lay, (0, 0), 3), (W, H))
        frame += 1.4 * lay
