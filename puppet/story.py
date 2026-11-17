"""Pieces for story clips (clips that are not a band show; see clips/hunt, clips/achzar):

  cuts(bars, a, b, cycle, n=1, opts=None)   # shots of n bars from a to b, taking turns in cycle
  Empty                                    # a cast member that draws nothing (the land alone)
  Rising(player, t0, t1, dy)               # a player coming up from below the frame
  Panels(render, panels, gutter)           # comic panels: several shots at once in frames on paper
  Duotone(...)                             # a look: posterized two-color print with ink lines
  silhouette(frame, mask, amount, rim)     # the cast as black cut-outs with a rim of light
  label(img, text)                         # a small caption in a corner (e.g. which experiment)
  Scope(render, shot, target, ...)         # a look through a rifle scope at another shot
  Falling(player, t0)                      # a player hit: tips back and drops out of the picture
  sfx(img, text, u, at)                    # comic lettering: "BANG!" popping in (u: seconds since)
  close_page(img, u)                       # the picture closes like a comic book (u: 0..1)
"""
import math

import cv2
import numpy as np

from production import H, ROOT, W


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

    def __init__(self, render, panels, t0, t1, paper=(0.86, 0.9, 0.93), border=7, stagger=0.35, tilt=0.0, dive=None):
        """dive: index of a panel the camera flies into over the shot's last 0.7 s (so the next
        shot, full screen, continues it)."""
        self.render, self.panels, self.t0, self.t1 = render, panels, t0, t1
        self.paper, self.border, self.stagger, self.tilt = np.array(paper, np.float32), border, stagger, tilt
        self.dive = dive

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
        if self.dive is not None and t > self.t1 - 0.7:  # fly into one panel until it fills the screen
            x, y, w, h = self.panels[self.dive][1]
            u = min(1.0, (t - (self.t1 - 0.7)) / 0.7)
            u = u * u * (3 - 2 * u)
            z = 1 + (max(W / w, H / h) - 1) * u
            cx, cy = (x + w / 2) * u + W / 2 * (1 - u), (y + h / 2) * u + H / 2 * (1 - u)
            M = np.float32([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]])
            frame[:] = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return frame


class Scope:
    """A look through a rifle scope: another shot (render(shot, t), like Panels) magnified around
    target(t) (screen xy), in a black circle with a crosshair; the aim drifts with breathing and
    settles; color: the reticle's (red reads as "he is the prey")."""
    j = {"credit": "Drawing: Ben"}

    def __init__(self, render, shot, target, mag=2.0, color=(0.05, 0.05, 0.05), seed=0, settle=None):
        self.render, self.shot, self.target, self.mag, self.color = render, shot, target, mag, np.array(color, np.float32)
        self.seed, self.settle = seed, settle
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = H * 0.46
        d = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        self.mask = np.clip((r - d) / 3.0, 0, 1)[..., None]
        self.vig = np.clip(1 - ((d / r) ** 4) * 0.55, 0, 1)[..., None]

    def draw(self, frame, t):
        tx, ty = self.target(t) if callable(self.target) else self.target
        k = 1.0 if self.settle is None else max(0.15, min(1.0, (self.settle - t) / 1.5))
        tx += k * (26 * math.sin(t * 1.3 + self.seed) + 9 * math.sin(t * 3.7))
        ty += k * (18 * math.sin(t * 1.1 + 2 + self.seed) + 7 * math.sin(t * 4.3))
        sub = self.render(self.shot, t)
        z = self.mag
        M = np.float32([[z, 0, W / 2 - z * tx], [0, z, H / 2 - z * ty]])
        img = cv2.warpAffine(sub, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = img * self.vig
        frame[:] = img * self.mask + np.array([0.01, 0.01, 0.01], np.float32) * (1 - self.mask)
        c = self.color
        cx, cy, r = W // 2, H // 2, int(H * 0.46)
        for (x0, y0, x1, y1, w) in ((cx - r, cy, cx - 40, cy, 3), (cx + 40, cy, cx + r, cy, 3), (cx, cy - r, cx, cy - 40, 3), (cx, cy + 40, cx, cy + r, 3),
                                    (cx - r, cy, cx - int(r * 0.7), cy, 9), (cx + int(r * 0.7), cy, cx + r, cy, 9),
                                    (cx, cy + int(r * 0.7), cx, cy + r, 9)):
            cv2.line(frame, (x0, y0), (x1, y1), tuple(float(v) for v in c), w, cv2.LINE_AA)
        for i in range(1, 4):  # range marks
            cv2.line(frame, (cx - 14, cy + 40 + i * 34), (cx + 14, cy + 40 + i * 34), tuple(float(v) for v in c), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, tuple(float(v) for v in c), -1, cv2.LINE_AA)
        return frame


class Falling:
    """A player who is hit at t0: a jolt, then tips back and drops out of the bottom of the picture
    over dur seconds; nothing after."""

    def __init__(self, p, t0, dur=1.2, lean=-14.0):
        self.p, self.t0, self.dur, self.lean = p, t0, dur, lean

    def __getattr__(self, k):
        return getattr(self.p, k)

    def draw(self, frame, t):
        u = (t - self.t0) / self.dur
        if u <= 0:
            return self.p.draw(frame, t)
        if u >= 1:
            return
        lay = np.zeros_like(frame)
        bg = frame.copy()
        self.p.draw(bg, min(t, self.t0 + 0.05))  # frozen at the hit
        m = (np.abs(bg - frame).max(2, keepdims=True) > 0.003).astype(np.float32)
        lay = bg
        e = u * u
        ang = self.lean * min(1.0, u * 3) + self.lean * 2 * e
        jolt = 18 * math.exp(-u * 12) * math.sin(u * 80)
        M = cv2.getRotationMatrix2D((W / 2, H * 1.1), ang, 1.0)
        M[0, 2] += jolt
        M[1, 2] += e * H * 0.9
        lay = cv2.warpAffine(lay * m, M, (W, H))
        mm = cv2.warpAffine(m, M, (W, H))[..., None]
        frame[:] = frame * (1 - mm) + lay


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


def sfx(img, text, u, at=(0.68, 0.3), size=150, angle=-8.0, fill=(40, 40, 225), burst=(70, 225, 255), dur=0.9):
    """Comic lettering on a finished uint8 BGR frame: `text` bursting in (u: seconds since the
    hit; gone after dur), a jagged burst behind it, black outline. at: center (share of W, H);
    fill, burst: BGR 0..255."""
    if not (0 <= u < dur):
        return img
    from PIL import Image, ImageDraw, ImageFont
    pop = min(1.0, u / 0.08)
    sc = (0.6 + 0.55 * pop) * (1 + 0.04 * math.sin(u * 40) * (1 - u / dur))
    fade = min(1.0, (dur - u) / 0.25)
    fs = int(size * sc)
    if fs not in _fonts:
        for p in (f"{ROOT}/public/fonts/bangers-latin-400-normal.woff",  # Bangers, SIL OFL (public/fonts/OFL-Bangers.txt)
                  "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                _fonts[fs] = ImageFont.truetype(p, fs)
                break
            except OSError:
                pass
    font = _fonts[fs]
    cx, cy = int(at[0] * W), int(at[1] * H)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    tw = d.textlength(text, font=font)
    rng = np.random.default_rng(len(text) * 7 + int(at[0] * 100))
    R = tw * 0.62 + 30
    pts = []
    for i in range(26):  # the burst
        a = 2 * math.pi * i / 26
        rr = R * (1.0 if i % 2 == 0 else 0.72) * rng.uniform(0.9, 1.1)
        pts.append((cx + rr * math.cos(a), cy + rr * 0.62 * math.sin(a)))
    d.polygon(pts, fill=tuple(burst[::-1]) + (240,), outline=(10, 10, 10, 255), width=6)
    d.text((cx - tw / 2, cy - fs * 0.62), text, font=font, fill=fill[::-1] + (255,), stroke_width=max(3, fs // 18), stroke_fill=(10, 10, 10, 255))
    lay = lay.rotate(angle, center=(cx, cy), resample=Image.BICUBIC)
    a = np.asarray(lay, np.float32)
    al = a[..., 3:4] / 255 * fade
    rgb = a[..., :3][..., ::-1]
    f = img.astype(np.float32)
    img[:] = np.clip(f * (1 - al) + rgb * al, 0, 255).astype(np.uint8)
    return img


def close_page(img, u):
    """The picture closes like a comic book: it folds shut from the right toward the spine at
    the left, darkening, onto black (u: 0..1)."""
    if u <= 0:
        return img
    u = min(1.0, u)
    e = u * u * (3 - 2 * u)
    w = max(1, int(W * (1 - e)))
    out = np.zeros_like(img)
    page = cv2.resize(img, (w, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    shade = np.linspace(1.0, 1.0 - 0.7 * e, w, dtype=np.float32)[None, :, None]
    out[:, :w] = np.clip(page * shade * (1 - 0.3 * e), 0, 255).astype(np.uint8)
    img[:] = out
    return img
