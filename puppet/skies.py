"""Night skies to put behind an open-sky plate (puppet/scenery3d.py clearing), drawn in code and
animated slowly. All are BGR float images, H x W, display space.

  sk = Skies(seed=5)
  img = sk.at("stars", t)          # "twilight", "stars", "clouds", "aurora"
  img = sk.mix(schedule, t)        # schedule: [(t_start, name), ...], crossfading at each change

- twilight: just after sunset: deep blue fading to a pink-orange glow low on one side, the first
  bright stars.
- stars: a dark sky with a Milky Way (a band of faint light and dust lanes), many stars of
  different colors, a faint airglow near the horizon; the stars twinkle.
- clouds: a full moon (seas, limb darkening, halo) behind drifting clouds, lit from the moon's side,
  stars in the gaps.
- aurora: green curtains with purple tops, rays moving slowly, over the stars.
Day and weather (the hunt clip):
- dawn: pale blue overhead, a warm glow and the sun's halo low on one side, thin pink clouds.
- day: clear blue with a few drifting cumulus.
- storm: low dark clouds rolling fast, lighter where they thin (lightning is weather.py's job).
- blood: a red-black sky, a big red moon, clouds lit red from below.
- ash: a brown-grey haze with smoke drifting up and a dim red sun.
"""
import math

import cv2
import numpy as np

from production import H, W

h2, w2 = H // 2, W // 2


def _fbm(rng, h, w, octaves=6, base=4, gain=0.55):
    """Fractal noise in [0, 1], tileable enough for slow drifts (generated a bit wider)."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.normal(0, 1, (max(2, int(n * h / w) + 2), n + 2)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= gain
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def _starfield(rng, n, h, w, yfrac=1.0, bias=None):
    """Points with a power-law brightness and star colors; bias(x, y) -> extra density weight."""
    xs = rng.uniform(0, w, n * 3)
    ys = rng.uniform(0, h * yfrac, n * 3)
    if bias is not None:
        keep = rng.uniform(0, 1, len(xs)) < bias(xs, ys)
        xs, ys = xs[keep][:n], ys[keep][:n]
    else:
        xs, ys = xs[:n], ys[:n]
    b = np.minimum(1.0, rng.pareto(2.0, len(xs)) * 0.06 + 0.03)
    temp = rng.uniform(0, 1, len(xs))  # blue-white to orange
    col = np.stack([0.9 + 0.1 * (1 - temp), 0.92 - 0.02 * temp, 0.8 + 0.2 * temp], 1)  # BGR
    return xs, ys, b, col, rng.uniform(0, 6.3, len(xs)), rng.uniform(1.5, 4.0, len(xs))


def _draw_stars(img, stars, t, tw=0.25):
    xs, ys, b, col, ph, sp = stars
    v = b * (1 - tw + tw * np.sin(t * sp + ph) ** 2)
    xi, yi = xs.astype(int), ys.astype(int)
    lay = np.zeros(img.shape, np.float32)
    np.add.at(lay, (yi, xi), (v[:, None] * col).astype(np.float32))
    lay = lay + cv2.GaussianBlur(lay, (0, 0), 0.9) * 2.0
    big = b > 0.35
    for x, y, c, vv in zip(xi[big], yi[big], col[big], v[big]):  # the brightest get a small glow
        cv2.circle(lay, (int(x), int(y)), 3, tuple(float(q) * 0.15 * float(vv) for q in c), -1, cv2.LINE_AA)
    img += lay


def _gradient(top, mid, low, h, w, hz=0.78):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None] / hz
    top, mid, low = (np.array(c, np.float32) for c in (top, mid, low))
    g = np.where(y < 0.55, top + (mid - top) * (y / 0.55), mid + (low - mid) * np.clip((y - 0.55) / 0.45, 0, 1))
    return np.ascontiguousarray(np.broadcast_to(g, (h, w, 3))).astype(np.float32)


class Skies:
    def __init__(self, seed=5):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.cache = {}
        # shared noise
        self.n1 = _fbm(rng, h2, w2 * 2, 7, 3)
        self.n2 = _fbm(rng, h2, w2 * 2, 6, 6)
        self.n3 = _fbm(rng, h2, w2, 5, 12)
        # the Milky Way: a band across the sky
        yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
        a = math.radians(-28)
        d = ((xx - w2 * 0.55) * math.sin(a) + (yy - h2 * 0.25) * math.cos(a)) / (h2 * 0.17)
        band = np.exp(-d ** 2)
        dust = np.clip((self.n3 - 0.45) * 3, 0, 1) * np.exp(-(d / 0.5) ** 2)
        glow = band * (0.35 + 0.65 * self.n1[:, :w2]) * (1 - 0.8 * dust)
        self.milky = glow[..., None] * np.array([0.26, 0.31, 0.36], np.float32)
        self.band_d = d
        self.stars_many = _starfield(rng, 9000, h2, w2, bias=lambda x, y: 0.25 + 0.75 * np.exp(-(((x - w2 * 0.55) * math.sin(a) + (y - h2 * 0.25) * math.cos(a)) / (h2 * 0.2)) ** 2))
        self.stars_few = _starfield(rng, 900, h2, w2)
        # the moon
        self.moon_xy, self.moon_r = (int(w2 * 0.74), int(h2 * 0.2)), 26
        r = self.moon_r
        my, mx = np.mgrid[-r - 2:r + 3, -r - 2:r + 3].astype(np.float32)
        rr = np.sqrt(mx ** 2 + my ** 2) / r
        disc = np.clip((1 - rr) * r, 0, 1)
        seas = _fbm(rng, 2 * r + 5, 2 * r + 5, 4, 3)
        tone = (0.95 - 0.35 * np.clip((seas - 0.5) * 3, 0, 1)) * (1 - 0.35 * np.clip(rr, 0, 1) ** 4)
        self.moon_img = (disc * tone)[..., None] * np.array([0.88, 0.95, 1.0], np.float32)
        self.moon_a = disc[..., None]
        md = np.sqrt((xx - self.moon_xy[0]) ** 2 + (yy - self.moon_xy[1]) ** 2)
        self.moon_halo = (np.exp(-md / 60) * 0.35 + np.exp(-md / 220) * 0.12)[..., None] * np.array([0.9, 0.85, 0.75], np.float32)
        self.moon_dist = md

    def _moon(self, img):
        x, y = self.moon_xy
        r = self.moon_r + 2
        sub = img[y - r:y + r + 1, x - r:x + r + 1]
        sub *= 1 - self.moon_a
        sub += self.moon_img

    def at(self, name, t):
        f = getattr(self, "_" + name)(t)
        return cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)

    def mix(self, schedule, t, fade=2.0):
        """schedule: [(t_start, name), ...] sorted; crossfade over `fade` s after each change."""
        cur = [s for s in schedule if s[0] <= t] or schedule[:1]
        t0, name = cur[-1]
        k = (t - t0) / fade
        img = self.at(name, t)
        if k < 1 and len(cur) > 1:
            img = img * k + self.at(cur[-2][1], t) * (1 - k)
        return img

    # ---- the skies (half size)
    def _twilight(self, t):
        g = _gradient((0.20, 0.07, 0.03), (0.40, 0.18, 0.12), (0.45, 0.32, 0.42), h2, w2)
        xx = np.linspace(0, 1, w2, dtype=np.float32)[None, :, None]
        yy = np.linspace(0, 1, h2, dtype=np.float32)[:, None, None]
        glow = np.exp(-((xx - 0.2) / 0.35) ** 2) * np.clip((yy - 0.35) / 0.45, 0, 1) ** 1.5
        g += glow * np.array([0.05, 0.22, 0.45], np.float32)
        _draw_stars(g, self.stars_few, t, tw=0.15)
        return g

    def _stars(self, t):
        g = _gradient((0.06, 0.025, 0.012), (0.09, 0.045, 0.025), (0.13, 0.12, 0.08), h2, w2)
        yy = np.linspace(0, 1, h2, dtype=np.float32)[:, None, None]
        g += np.clip((yy - 0.45) / 0.4, 0, 1) ** 2.5 * np.array([0.012, 0.03, 0.016], np.float32)  # airglow
        g += self.milky
        _draw_stars(g, self.stars_many, t)
        return g

    def _clouds(self, t):
        g = _gradient((0.16, 0.08, 0.04), (0.26, 0.16, 0.10), (0.30, 0.22, 0.16), h2, w2)  # a moonlit sky is brighter
        _draw_stars(g, self.stars_few, t)
        g += self.moon_halo
        self._moon(g)
        off = int(t * 9) % w2  # drifting
        c = np.roll(self.n2, -off, axis=1)[:, :w2] * 0.65 + np.roll(self.n1, -off // 2, axis=1)[:, :w2] * 0.35
        yy = np.linspace(0, 1, h2, dtype=np.float32)[:, None]
        lo, hi = np.percentile(c, 45), np.percentile(c, 80)
        cov = np.clip((c - lo) / (hi - lo), 0, 1) * np.clip(1.25 - yy * 0.5, 0, 1)
        c = np.clip((c - lo) / (hi - lo + 1e-6), 0, 1)
        lit = np.exp(-self.moon_dist / 420) * 0.9 + 0.2
        thin = np.exp(-self.moon_dist / 110) * (1 - c)  # thin edges near the moon glow
        col = lit[..., None] * np.array([0.34, 0.29, 0.26], np.float32) * (0.55 + 0.45 * (1 - c[..., None])) \
            + thin[..., None] * np.array([0.55, 0.58, 0.6], np.float32)
        cov = cv2.GaussianBlur(cov, (0, 0), 2.0) ** 0.8
        a = cov[..., None] * 0.92
        return g * (1 - a) + col * a

    def _aurora(self, t):
        g = self._stars(t) * 0.8
        x = np.arange(w2, dtype=np.float32)
        base = h2 * 0.42 + 34 * np.sin(x / 140 + t * 0.15) + 16 * np.sin(x / 57 - t * 0.23)  # the curtain's lower edge
        yy = np.arange(h2, dtype=np.float32)[:, None]
        up = base[None, :] - yy  # px above the edge
        env = np.where(up >= 0, np.exp(-np.maximum(up, 0) / 170), np.exp(np.minimum(up, 0) / 4))
        off = int(t * 14) % (w2 * 2)
        rays = np.roll(self.n2, -off, axis=1)[:, :w2].mean(0)  # vertical rays: noise along x only
        rays = 0.35 + 0.65 * np.clip((rays - rays.min()) / (rays.max() - rays.min() + 1e-9), 0, 1) ** 1.3
        fold = 0.6 + 0.4 * np.sin(x / 33 + t * 0.6 + np.sin(x / 200) * 3)
        inten = env * (rays * fold)[None, :] * np.clip(up / 8 + 1, 0, 1)
        k = np.clip(up / 160, 0, 1)
        color = (1 - k)[..., None] * np.array([0.35, 1.0, 0.45], np.float32) + k[..., None] * np.array([0.6, 0.25, 0.55], np.float32)
        return g + inten[..., None] * color * 0.8

    # ---- day and weather skies (half size, BGR)
    def _cloud_field(self, t, speed, lo_p, hi_p, mix=0.65):
        off = int(t * speed) % w2
        c = np.roll(self.n2, -off, axis=1)[:, :w2] * mix + np.roll(self.n1, -off // 2, axis=1)[:, :w2] * (1 - mix)
        lo, hi = np.percentile(c, lo_p), np.percentile(c, hi_p)
        return np.clip((c - lo) / (hi - lo + 1e-6), 0, 1)

    def _sun(self, img, x, y, r, col, halo=1.0):
        yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        img += (np.exp(-d / (r * 3)) * 0.5 * halo + np.exp(-d / (r * 12)) * 0.25 * halo)[..., None] * np.array(col, np.float32)
        disc = np.clip(r - d, 0, 1)[..., None]
        img[:] = np.maximum(img, disc * np.minimum(np.array(col, np.float32) * 1.8, 1.0))

    def _dawn(self, t):
        g = _gradient((0.62, 0.42, 0.28), (0.66, 0.60, 0.62), (0.55, 0.72, 0.98), h2, w2, hz=0.82)
        self._sun(g, w2 * 0.22, h2 * 0.74, 14, (0.55, 0.80, 1.0))
        c = self._cloud_field(t, 3, 60, 92)
        yy = np.linspace(0, 1, h2, dtype=np.float32)[:, None]
        a = (c * np.clip(1.0 - np.abs(yy - 0.35) * 2.2, 0, 1))[..., None] * 0.7
        return g * (1 - a) + a * np.array([0.62, 0.62, 0.92], np.float32)

    def _day(self, t):
        g = _gradient((0.78, 0.48, 0.22), (0.86, 0.66, 0.45), (0.92, 0.84, 0.78), h2, w2, hz=0.85)
        c = self._cloud_field(t, 6, 62, 85)
        yy = np.linspace(0, 1, h2, dtype=np.float32)[:, None]
        a = (c * np.clip(1.1 - yy * 0.9, 0, 1))[..., None] * 0.9
        shade = 0.82 + 0.18 * (1 - c[..., None])
        return g * (1 - a) + a * np.array([0.97, 0.97, 0.98], np.float32) * shade

    def _storm(self, t):
        g = _gradient((0.16, 0.15, 0.14), (0.22, 0.21, 0.20), (0.28, 0.28, 0.27), h2, w2)
        c = self._cloud_field(t, 40, 20, 90, mix=0.5)
        c2 = self._cloud_field(t * 1.7 + 50, 70, 30, 95, mix=0.8)
        dark = 0.55 * c + 0.45 * c2
        col = np.array([0.30, 0.29, 0.28], np.float32) * (1.15 - 0.85 * dark[..., None])
        return g * 0.3 + col

    def _blood(self, t):
        g = _gradient((0.03, 0.02, 0.10), (0.05, 0.04, 0.30), (0.10, 0.12, 0.62), h2, w2, hz=0.8)
        yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
        mx, my, r = w2 * 0.66, h2 * 0.3, 58
        d = np.sqrt((xx - mx) ** 2 + (yy - my) ** 2)
        g += (np.exp(-d / 90) * 0.5 + np.exp(-d / 300) * 0.2)[..., None] * np.array([0.05, 0.10, 0.75], np.float32)
        disc = np.clip(r - d, 0, 1)[..., None]
        seas = cv2.resize(self.n3, (w2, h2))
        moon = np.array([0.10, 0.16, 0.85], np.float32) * (0.8 + 0.3 * seas[..., None])
        g = g * (1 - disc) + moon * disc
        c = self._cloud_field(t, 8, 50, 88)
        a = (c * np.clip((yy / h2 - 0.15) * 1.6, 0, 1))[..., None] * 0.85
        lit = np.array([0.04, 0.06, 0.45], np.float32) * (0.6 + 0.8 * (yy / h2)[..., None])
        return g * (1 - a) + lit * a

    def _ash(self, t):
        g = _gradient((0.20, 0.22, 0.26), (0.22, 0.26, 0.33), (0.20, 0.27, 0.40), h2, w2)
        self._sun(g, w2 * 0.6, h2 * 0.38, 20, (0.12, 0.20, 0.60), halo=0.6)
        n = self.n1[:, :w2]
        n = np.concatenate([n, n[::-1]], 0)  # mirrored: rolls without a seam
        sm = np.roll(n, -int(t * 16) % (2 * h2), axis=0)[:h2]  # smoke rising
        a = np.clip((sm - 0.4) * 2.2, 0, 1)[..., None] * 0.6
        return g * (1 - a) + a * np.array([0.13, 0.14, 0.16], np.float32)
