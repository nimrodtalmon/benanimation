"""Make 3D renders look drawn, so they sit with Ben's drawings: one style for everything 3D (the kit,
guitars, close-ups, scenery, fire). A 2D filter on finished renders, so any source can use it.

  ink.use("ben")                     # once per clip (None: leave renders as they are)
  out = ink.draw(img, alpha=None)    # BGR float [0,1] (+ alpha) -> drawn BGR (+ alpha grown by the outline)
  out = ink.cached(path_or_key, img, alpha)   # the same, cached on disk (static layers, close-up frames)
  ink.Lines / ink.flat               # the two halves, for sources relit per frame (prepare lines once)

Style "ben": slightly wobbly dark-gray pencil lines from the picture's edges (eXtended Difference of
Gaussians on a smoothed picture, plus the silhouette), colors simplified (edge-preserving smoothing,
lightness pulled into a few soft bands, a little lighter and less saturated, like colored pencil on
white paper), pencil strokes in the darker bands. The paper itself comes from the global look
(production.HandDrawn).
"""
import hashlib
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLES = {
    "ben": dict(smooth=(9, 40), bands=4, soft=0.35, lighten=0.18, desat=0.12, line=(0.13, 0.12, 0.12), line_w=1.0,
                xdog=(1.0, 1.6, 0.985, 0.012, 60.0), wobble=1.6, strokes=0.18, sil_w=2.2),
}
_style = None


def use(name):
    """Choose the style for this run (None: no stylizing). Also tells the Blender builders (through
    the environment, inherited by their processes) to draw outlines from the geometry."""
    global _style
    _style = None if name is None else dict(STYLES[name], name=name)
    if name is None:
        os.environ.pop("BENANIM_STYLE", None)
    else:
        os.environ["BENANIM_STYLE"] = name


def active():
    return _style is not None


def _noise(h, w, scale, seed):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, (max(2, h // scale + 2), max(2, w // scale + 2))).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _strokes(h, w, seed=3):
    """Pencil strokes: noise stretched along a slant."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (h + w, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=0.6, sigmaY=9)
    M = cv2.getRotationMatrix2D((w / 2, (h + w) / 2), 35, 1.0)
    n = cv2.warpAffine(n, M, (w, h + w))[:h]
    return (n - n.mean()) / (n.std() + 1e-6)


def flat(img, st=None, scale=1.0):
    """Simplified colors: edge-preserving smoothing, lightness in soft bands, lighter, a bit paler."""
    st = st or _style
    d, sc = st["smooth"]
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    sm = cv2.bilateralFilter(u8, max(3, int(d * scale)) | 1, sc, sc).astype(np.float32) / 255
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB)
    L = lab[..., 0] / 100.0
    k = st["bands"]
    q = np.floor(L * k) / k
    frac = L * k - np.floor(L * k)
    Lq = q + np.clip((frac - (1 - st["soft"])) / st["soft"], 0, 1) / k  # soft steps between bands
    Lq = Lq * (1 - st["lighten"]) + st["lighten"]
    lab[..., 0] = np.clip(Lq, 0, 1) * 100
    lab[..., 1:] *= 1 - st["desat"]
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    h, w = img.shape[:2]
    if st["strokes"]:
        s = _strokes(h, w)
        dark = np.clip(1 - Lq, 0, 1)[..., None]
        out = out * (1 - st["strokes"] * dark * np.clip(0.5 + 0.5 * s, 0, 1)[..., None])
    return np.clip(out, 0, 1)


class Lines:
    """Pencil lines for a picture (and its silhouette), made once; draw(out) puts them on."""

    def __init__(self, img, alpha=None, st=None, scale=1.0):
        st = st or _style
        h, w = img.shape[:2]
        g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        g = cv2.bilateralFilter((g * 255).astype(np.uint8), 7, 30, 30).astype(np.float32) / 255
        s1, s2, tau, eps, phi = st["xdog"]
        s1, s2 = s1 * scale, s2 * scale
        dog = cv2.GaussianBlur(g, (0, 0), s1) - tau * cv2.GaussianBlur(g, (0, 0), s2)
        lines = np.where(dog < eps, 1 - np.tanh(phi * (dog - eps)), 1.0)  # 1 = paper, lower = ink
        ink = np.clip(1 - lines, 0, 1)
        if alpha is not None:
            a = np.clip(alpha.reshape(h, w), 0, 1)
            k = max(1, int(round(st["sil_w"] * scale)))
            edge = cv2.dilate(a, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) - cv2.erode(a, np.ones((3, 3), np.uint8))
            ink = np.maximum(ink * (a > 0.05), np.clip(edge, 0, 1))
        wob = st["wobble"] * scale
        if wob:
            mx = np.arange(w, dtype=np.float32)[None, :] + wob * _noise(h, w, 40, 1)
            my = np.arange(h, dtype=np.float32)[:, None] + wob * _noise(h, w, 40, 2)
            ink = cv2.remap(ink, mx, my, cv2.INTER_LINEAR)
        tex = 0.75 + 0.25 * np.clip(0.5 + 0.5 * _noise(h, w, 3, 4), 0, 1)  # pencil grain along the line
        self.ink = np.clip(ink * tex * st["line_w"], 0, 1)[..., None]
        self.color = np.array(st["line"], np.float32)
        self.alpha = None
        if alpha is not None:  # the outline grows the coverage a little
            self.alpha = np.maximum(np.clip(alpha.reshape(h, w, 1), 0, 1), self.ink * (self.ink > 0.3))

    def draw(self, out):
        return out * (1 - self.ink) + self.color * self.ink


def draw(img, alpha=None, st=None, scale=1.0):
    """img BGR float (h, w, 3); alpha (h, w) or (h, w, 1) or None. Returns BGR, or BGRA if alpha."""
    st = st or _style
    if st is None:
        return img if alpha is None else np.concatenate([img, alpha.reshape(img.shape[0], img.shape[1], 1)], 2)
    L = Lines(img, alpha, st, scale)
    out = L.draw(flat(img, st, scale))
    if alpha is None:
        return out.astype(np.float32)
    return np.concatenate([out, L.alpha], 2).astype(np.float32)


def lookup(path, scale=1.0):
    """The flattened version of an image file if it is already cached (no need to read the file)."""
    if _style is None or not os.path.exists(path):
        return None
    k = f"('flat', '{path}'):{os.path.getmtime(path)}|{_style}|{scale}|file"
    p = f"{ROOT}/out/inkcache/{hashlib.sha1(k.encode()).hexdigest()[:16]}.png"
    return cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 if os.path.exists(p) else None


def store(path, out, scale=1.0):
    k = f"('flat', '{path}'):{os.path.getmtime(path)}|{_style}|{scale}|file"
    p = f"{ROOT}/out/inkcache/{hashlib.sha1(k.encode()).hexdigest()[:16]}.png"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, (np.clip(out, 0, 1) * 255).astype(np.uint8))


def flat_file(path, scale=1.0):
    """An image file (RGBA PNG) with flattened colors, cached by path: reads only the cache when it can."""
    out = lookup(path, scale)
    if out is not None:
        return out
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
    if _style is None:
        return im
    out = flat(im[..., :3], scale=scale)
    if im.shape[2] == 4:
        out = np.concatenate([out, im[..., 3:4]], 2)
    store(path, out, scale)
    return out


def flat_cached(key, img, alpha=None, scale=1.0):
    """Only the colors (the outlines are already drawn by the builder), cached like cached()."""
    if _style is None:
        return img if alpha is None else np.concatenate([img, alpha.reshape(img.shape[0], img.shape[1], 1)], 2)
    return cached(("flat", key), img, alpha, scale, lines=False)


def cached(key, img, alpha=None, scale=1.0, lines=True):
    """draw() cached on disk under out/inkcache/ (key: a file path with its mtime, or any string)."""
    if _style is None:
        return draw(img, alpha)
    k0 = key[1] if isinstance(key, tuple) else key
    if os.path.exists(str(k0)):
        key = f"{key}:{os.path.getmtime(k0)}"
    h = hashlib.sha1(f"{key}|{_style}|{scale}|{img.shape}|{lines}".encode()).hexdigest()[:16]
    p = f"{ROOT}/out/inkcache/{h}.png"
    if os.path.exists(p):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        return im
    if lines:
        out = draw(img, alpha, scale=scale)
    else:
        out = flat(img, scale=scale)
        if alpha is not None:
            out = np.concatenate([out, alpha.reshape(img.shape[0], img.shape[1], 1)], 2)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, (np.clip(out, 0, 1) * 255).astype(np.uint8))
    return out
