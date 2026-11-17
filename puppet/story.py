"""Pieces for story clips (clips that are not a band show; see clips/hunt, clips/achzar):

  cuts(bars, a, b, cycle, n=1, opts=None)   # shots of n bars from a to b, taking turns in cycle
  Empty                                    # a cast member that draws nothing (the land alone)
  Rising(player, t0, t1, dy)               # a player coming up from below the frame
  Panels(render, panels, gutter)           # comic panels: several shots at once in frames on paper
  Duotone(...)                             # a look: posterized two-color print with ink lines
  silhouette(frame, mask, amount, rim)     # the cast as black cut-outs with a rim of light
  label(img, text)                         # a small caption in a corner (e.g. which experiment)
"""
import math

import cv2
import numpy as np

from production import H, W


def cuts(bars, a, b, cycle, n=1, opts=None):
    """Shots of n bars from a to b, taking turns in cycle; an entry may be a callable(a, b) -> who,
    opts a dict or a callable(a, b) -> dict. Returns [(t0, t1, who, opts)]."""
    out = []
    i = int(np.searchsorted(bars, a - 0.05))
    k, t0 = 0, a
    while t0 < b - 0.05:
        j = min(i + n, len(bars) - 1)
        t1 = min(b, bars[j]) if bars[j] > t0 + 0.2 else b
        who = cycle[k % len(cycle)]
        if callable(who):
            who = who(t0, t1)
        o = opts(t0, t1) if callable(opts) else dict(opts or {})
        out.append((t0, t1, who, o))
        k += 1
        t0, i = t1, j
    return out


def beat_cuts(beats, a, b, cycle, n=2, opts=None):
    """Like cuts(), but n beats a shot."""
    return cuts(beats, a, b, cycle, n, opts)


class Empty:
    """No one in the picture."""
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


class Panels:
    """Comic panels: several shots drawn at once, each in its own frame on paper. render(shot, t)
    gives a full frame (H x W x 3 float) of a shot; panels: [(shot, (x, y, w, h), zoom)] in screen
    px (each shot is cropped to its panel's shape, slowly pushing in by zoom over the shot); the
    panels slide in one after another from t0 (every `stagger` s)."""
    j = {"credit": "Drawing: Ben"}

    def __init__(self, render, panels, t0, t1, paper=(0.86, 0.9, 0.93), border=7, stagger=0.35, tilt=0.0):
        self.render, self.panels, self.t0, self.t1 = render, panels, t0, t1
        self.paper, self.border, self.stagger, self.tilt = np.array(paper, np.float32), border, stagger, tilt

    def draw(self, frame, t):
        frame[:] = self.paper
        k = (t - self.t0) / max(1e-6, self.t1 - self.t0)
        for i, (shot, (x, y, w, h), zm) in enumerate(self.panels):
            u = (t - self.t0 - i * self.stagger) / 0.3
            if u <= 0:
                continue
            u = min(1.0, u)
            e = 1 - (1 - u) ** 3
            sub = self.render(shot, t)
            z = 1.0 + zm * k
            ar = w / h
            ch = H / z
            cw = ch * ar
            if cw > W / z:
                cw = W / z
                ch = cw / ar
            cx, cy = W / 2, H * 0.45
            x0, y0 = int(max(0, min(W - cw, cx - cw / 2))), int(max(0, min(H - ch, cy - ch / 2)))
            crop = sub[y0:y0 + int(ch), x0:x0 + int(cw)]
            img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            dx = int((1 - e) * (W * 0.6) * (-1 if i % 2 else 1))  # sliding in
            X, Y = x + dx, y
            xa, xb = max(0, X), min(W, X + w)
            if xb <= xa:
                continue
            b = self.border
            cv2.rectangle(frame, (xa - b, Y - b), (xb + b - 1, Y + h + b - 1), (0.04, 0.04, 0.05), -1)
            frame[Y:Y + h, xa:xb] = img[:, xa - X:xb - X]
        return frame


class Duotone:
    """A look: the picture printed in two inks on paper: dark ink, a color ink and the paper,
    posterized into a few flat steps, with ink lines on the edges (a graphic-novel page)."""

    def __init__(self, dark=(0.06, 0.03, 0.04), ink=(0.10, 0.12, 0.72), paper=(0.80, 0.88, 0.93), steps=4, lines=0.8, boil=0.6, hold=2):
        self.dark, self.ink, self.paper = (np.array(c, np.float32) for c in (dark, ink, paper))
        self.steps, self.lines = steps, lines
        from production import HandDrawn
        self.hd = HandDrawn(boil=boil, hold=hold, paper=0.10, grade=0.0, seed=3)

    def __call__(self, frame, f):
        lum = cv2.cvtColor(np.clip(frame, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
        lum = cv2.bilateralFilter(lum, 7, 0.1, 5)
        lo, hi = np.percentile(lum, 3), np.percentile(lum, 97)
        x = np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1)
        q = np.floor(x * self.steps) / (self.steps - 1)  # flat steps
        sm = x * self.steps - np.floor(x * self.steps)
        q = np.clip(q + (np.clip((sm - 0.85) / 0.15, 0, 1)) / (self.steps - 1), 0, 1)  # soft step edges
        q = q[..., None]
        out = np.where(q < 0.5, self.dark + (self.ink - self.dark) * (q * 2), self.ink + (self.paper - self.ink) * (q * 2 - 1))
        if self.lines > 0:
            g = cv2.GaussianBlur(lum, (0, 0), 1.2)
            e = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
            e = np.clip((e - 0.02) / 0.06, 0, 1)[..., None] * self.lines
            out = out * (1 - e) + self.dark * e
        return self.hd(out.astype(np.float32), f)


EMIT = []  # light a cast member gives off (glowing eyes...): callables(frame) drawn again over silhouettes


def emit(fn):
    """A cast member registers light it gives off this frame (fn(frame) draws it); silhouette()
    draws these again on top, so glowing eyes shine out of a black cut-out."""
    EMIT.append(fn)


def silhouette(frame, mask, amount=1.0, rim=(0.25, 0.6, 1.0), dark=(0.02, 0.015, 0.02), side=(1, -1)):
    """The cast (mask: H x W x 1, 0..1) as black cut-outs with a rim of light on one side; the
    light they give off (emit()) shines through."""
    if amount <= 0:
        return frame
    lum = frame.mean(2, keepdims=True)
    keep = np.clip((lum - 0.97) / 0.03, 0, 1)
    m = mask * (1 - keep) * amount
    m2 = mask[..., 0]
    sh = np.roll(np.roll(m2, -side[0] * 5, axis=1), -side[1] * 5, axis=0)  # the mask moved toward the light
    r = np.clip(m2 - sh, 0, 1)
    r = cv2.GaussianBlur(r, (0, 0), 1.5)[..., None] * amount
    frame[:] = frame * (1 - m) + m * np.array(dark, np.float32)
    frame += r * np.array(rim, np.float32)
    np.minimum(frame, 1.0, out=frame)
    for fn in EMIT:
        fn(frame)
    EMIT.clear()
    return frame


_fonts = {}


def label(img, text, size=34, at=(36, 40), alpha=0.85):
    """A small caption on a finished uint8 BGR frame (in place): white text with a soft shadow."""
    from PIL import Image, ImageDraw, ImageFont
    if size not in _fonts:
        for p in ("/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                _fonts[size] = ImageFont.truetype(p, size)
                break
            except OSError:
                pass
    font = _fonts.get(size) or ImageFont.load_default()
    lay = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lay)
    d.text(at, text, fill=255, font=font)
    m = np.asarray(lay, np.float32)[..., None] / 255 * alpha
    sh = cv2.GaussianBlur(np.roll(np.roll(m[..., 0], 3, 0), 3, 1), (0, 0), 3)[..., None]
    f = img.astype(np.float32)
    f = f * (1 - 0.6 * sh) + 0 * sh
    f = f * (1 - m) + 245 * m
    img[:] = np.clip(f, 0, 255).astype(np.uint8)
    return img


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def slow_pan(a, b, dx=160, y=H * 0.42):
    """focus(t) for a long take: a slow pan across the shot."""
    return lambda t: (W / 2 + dx * (ease((t - a) / (b - a)) - 0.5), y + 10 * math.sin(t * 0.4))
