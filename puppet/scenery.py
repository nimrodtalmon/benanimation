"""Stage additions: a band logo cut from Ben's page (as a lit sign on the back wall, and as a title
card), floor haze, and helpers for light cues on the song's breaks.

  logo = ben_logo()                          # alpha mask of the hand-lettered band name
  sign = Sign(logo, width=900, y=170)        # sign.draw(frame, t, color, k) into the stage
  haze = Haze(); haze.draw(frame, t, k)      # slow drifting floor smoke
  title_card(frame_u8, logo, k)              # the logo over black, for openings and endings
"""
import math
import os

import cv2
import numpy as np

import photo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1920, 1080


def ben_logo(src="drawings/ben/dr/band-sheet.jpg", box=(470, 214, 1190, 582), cut=(), cache="public/parts/logo.png"):
    """The band name as Ben lettered it on the band page: ink -> alpha (0..1). box = x0, y0, x1, y1
    on the light-flattened page; cut = boxes to erase (the drawing below the letters)."""
    p = f"{ROOT}/{cache}"
    if os.path.exists(p):
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    im = photo.flat_light(cv2.imread(f"{ROOT}/{src}"))
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = np.clip((170 - g) / 90, 0, 1)
    for (cx0, cy0, cx1, cy1) in cut:
        a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = 0
    # keep the letters (big ink blobs), drop specks
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.4).astype(np.uint8))
    keep = np.zeros(n, bool)
    keep[1:] = (st[1:, cv2.CC_STAT_AREA] > 60) & (st[1:, cv2.CC_STAT_WIDTH] < 2.5 * st[1:, cv2.CC_STAT_HEIGHT])  # no long flat strokes (the head drawn below)
    a *= cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    cv2.imwrite(p, (a * 255).astype(np.uint8))
    return a


class Sign:
    """The logo as a glowing sign hung on the back wall (drawn into the stage before the players,
    so it gets the wall's parallax). The glow takes the color it is given (the light of the moment)."""

    def __init__(self, logo, width=900, cx=W / 2, y=150):
        h = int(logo.shape[0] * width / logo.shape[1])
        m = cv2.resize(logo, (width, h), interpolation=cv2.INTER_AREA)
        self.core = np.zeros((H, W), np.float32)
        x0 = int(cx - width / 2)
        self.core[y:y + h, x0:x0 + width] = m
        self.glow = cv2.GaussianBlur(self.core, (0, 0), 14) * 1.6
        self.halo = cv2.GaussianBlur(self.core, (0, 0), 60) * 3.0

    def draw(self, frame, t, color=(0.9, 0.95, 1.0), k=1.0):
        flick = 1.0 - 0.04 * (math.sin(t * 37.0) > 0.97)  # a sign's tiny flicker
        c = np.array(color, np.float32)
        frame += (self.halo[..., None] * 0.12 + self.glow[..., None] * 0.35) * c * k * flick
        core = self.core[..., None] * k * flick
        frame[:] = frame * (1 - core) + core * (0.55 + 0.45 * c)


class Haze:
    """Slow floor smoke: a few soft layers drifting sideways, thicker near the floor."""

    def __init__(self, seed=2):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(3):
            g = rng.normal(0, 1, (12, 48)).astype(np.float32)
            g = cv2.resize(cv2.GaussianBlur(g, (0, 0), 1.2), (W * 2, H // 2), interpolation=cv2.INTER_CUBIC)
            g = np.clip((g - g.mean()) / (3 * g.std()) + 0.5, 0, 1)
            self.layers.append((np.tile(g, (1, 1)), 12 + 9 * i, rng.uniform(0, W)))
        yy = np.linspace(0, 1, H // 2, dtype=np.float32)[:, None]
        self.ramp = np.clip((yy - 0.25) / 0.75, 0, 1) ** 1.5

    def draw(self, frame, t, k=1.0, color=(0.75, 0.72, 0.8)):
        acc = np.zeros((H // 2, W), np.float32)
        for g, v, off in self.layers:
            x = int((off + v * t) % W)
            acc += g[:, x:x + W]
        acc = acc / len(self.layers) * self.ramp
        frame[H // 2:] += (acc[..., None] * 0.22 * k) * np.array(color, np.float32)


def title_card(img, logo, k, sub=None, y=None):
    """The logo, white over black, faded in by k (0..1): over a uint8 BGR frame (in place)."""
    if k <= 0:
        return img
    w = int(W * 0.62)
    h = int(logo.shape[0] * w / logo.shape[1])
    m = cv2.resize(logo, (w, h), interpolation=cv2.INTER_AREA)
    y = int(H * 0.5 - h * 0.6) if y is None else y
    x = (W - w) // 2
    f = img.astype(np.float32) / 255
    f *= 1 - 0.92 * k  # darken the picture
    glow = np.zeros((H, W), np.float32)
    glow[y:y + h, x:x + w] = m
    halo = cv2.GaussianBlur(glow, (0, 0), 18)
    f += (halo[..., None] * 0.5 * k) * np.array([0.55, 0.65, 0.95], np.float32)
    f = f * (1 - glow[..., None] * k) + glow[..., None] * k * 0.97
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    if sub:
        for i, line in enumerate(sub):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            c = int(220 * k)
            cv2.putText(out, line, ((W - tw) // 2, y + h + 70 + 44 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (c, c, c), 2, cv2.LINE_AA)
    img[:] = out
    return img
