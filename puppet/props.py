"""Props drawn in code in a marker style (black ink outline, flat tones, a highlight):
microphone, electric guitar, drum kit. Each returns a BGRA float image plus anchor points,
drawn at a fixed size; callers scale and rotate them.
"""
import math

import cv2
import numpy as np

INK = (0.075, 0.07, 0.08)


def _canvas(w, h):
    return np.zeros((h, w, 4), np.float32)


def _poly(img, pts, color, ink=7, ss=4):
    """Antialiased filled polygon with an ink outline (supersampled masks)."""
    h, w = img.shape[:2]
    P = np.round(np.asarray(pts, np.float64) * ss * 8).astype(np.int32)
    m = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(m, [P], 255, cv2.LINE_AA, shift=3)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    col = np.asarray(color, np.float32)
    if col.ndim == 1:
        col = col[None, None, :]
    img[..., :3] = img[..., :3] * (1 - m) + col * m
    img[..., 3:4] = np.maximum(img[..., 3:4], m)
    if ink:
        _line(img, pts, ink, closed=True)
    return m[..., 0]


def _line(img, pts, width, closed=False, color=INK, ss=4):
    h, w = img.shape[:2]
    P = np.round(np.asarray(pts, np.float64) * ss * 8).astype(np.int32)
    m = np.zeros((h * ss, w * ss), np.uint8)
    cv2.polylines(m, [P], closed, 255, max(1, int(width * ss)), cv2.LINE_AA, shift=3)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    img[..., :3] = img[..., :3] * (1 - m) + np.asarray(color, np.float32) * m
    img[..., 3:4] = np.maximum(img[..., 3:4], m)


def _ellipse_pts(cx, cy, rx, ry, a0=0, a1=2 * math.pi, n=80, rot=0.0):
    t = np.linspace(a0, a1, n)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def _gradient(h, w, c0, c1, axis=1, p=1.0):
    t = np.linspace(0, 1, w if axis == 1 else h, dtype=np.float32) ** p
    g = np.asarray(c0, np.float32) * (1 - t[:, None]) + np.asarray(c1, np.float32) * t[:, None]
    return np.broadcast_to(g[None] if axis == 1 else g[:, None], (h, w, 3)).copy()


def microphone(L=330):
    """Vertical handheld mic, grille on top. Returns (rgba, anchors: grip, head)."""
    w, h = 140, L + 40
    img = _canvas(w, h)
    cx = w / 2
    head_r = 50
    hy = 20 + head_r
    # handle: tapered, dark, with a highlight stripe
    top_w, bot_w = 30, 19
    y0, y1 = hy + head_r + 22, h - 14
    handle = [(cx - top_w, y0), (cx + top_w, y0), (cx + bot_w, y1), (cx - bot_w, y1)]
    _poly(img, handle, _gradient(h, w, (0.10, 0.10, 0.11), (0.30, 0.30, 0.33), p=0.8), ink=6)
    _line(img, [(cx + 10, y0 + 12), (cx + 6, y1 - 12)], 5, color=(0.55, 0.55, 0.58))
    # collar between handle and grille
    _poly(img, [(cx - 40, hy + head_r - 6), (cx + 40, hy + head_r - 6), (cx + 33, y0 + 4), (cx - 33, y0 + 4)],
          _gradient(h, w, (0.55, 0.55, 0.58), (0.85, 0.85, 0.88)), ink=6)
    # grille ball with a mesh and a highlight
    m = _poly(img, _ellipse_pts(cx, hy, head_r, head_r), _gradient(h, w, (0.35, 0.36, 0.38), (0.75, 0.76, 0.78), p=0.7), ink=0)
    mesh = np.zeros((h, w), np.float32)
    for k in range(-head_r, head_r, 11):
        cv2.line(mesh, (int(cx + k - 60), int(hy - 60)), (int(cx + k + 60), int(hy + 60)), 1.0, 2, cv2.LINE_AA)
        cv2.line(mesh, (int(cx + k + 60), int(hy - 60)), (int(cx + k - 60), int(hy + 60)), 1.0, 2, cv2.LINE_AA)
    mesh *= m * 0.45
    img[..., :3] *= (1 - mesh[..., None])
    hl = np.zeros((h, w), np.float32)
    cv2.ellipse(hl, (int(cx + 16), int(hy - 18)), (14, 9), -30, 0, 360, 1.0, -1, cv2.LINE_AA)
    hl = cv2.GaussianBlur(hl, (0, 0), 3) * m
    img[..., :3] = img[..., :3] * (1 - 0.8 * hl[..., None]) + 0.8 * hl[..., None]
    _line(img, _ellipse_pts(cx, hy, head_r, head_r), 6, closed=True)
    _line(img, [(cx - head_r + 6, hy + 4), (cx + head_r - 6, hy + 4)], 4)  # seam across the grille
    return img, dict(grip=(cx, (y0 + y1) / 2 - 15), head=(cx, hy))


def place(prop, anchor, world, angle, scale):
    """Affine that puts prop's anchor at world, rotated by angle (radians, 0 = as drawn)."""
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    R = np.array([[c, -s], [s, c]])
    t = np.asarray(world, float) - R @ np.asarray(anchor, float)
    return np.hstack([R, t[:, None]])


def blit(dst, prop, M):
    """Warp a BGRA prop with affine M and composite it over dst (BGR float), in place."""
    h, w = prop.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 1
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, dst.shape[1]), min(y1, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy()
    M2[:, 2] -= [x0, y0]
    lay = cv2.warpAffine(prop, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)
    a = lay[..., 3:4]
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + lay[..., :3] * a


# ---------------------------------------------------------------- drum kit
RED_SHELL = ((0.10, 0.06, 0.32), (0.25, 0.18, 0.68))  # BGR dark, light: deep red sparkle lacquer
CHROME = ((0.38, 0.38, 0.40), (0.92, 0.92, 0.94))
HEAD = (0.86, 0.90, 0.92)
BRASS = ((0.10, 0.38, 0.55), (0.45, 0.80, 0.95))


def _shell_tone(h, w, dark, light):
    """Cylinder shading across x: dark at the sides, light stripe left of center."""
    x = np.linspace(-1, 1, w, dtype=np.float32)
    k = np.clip(1 - np.abs(x + 0.25) ** 1.4, 0, 1)
    col = np.asarray(dark, np.float32) * (1 - k[:, None]) + np.asarray(light, np.float32) * k[:, None]
    return np.broadcast_to(col[None], (h, w, 3)).copy()


def _sparkle_shell(h, w, dark, light, seed=0, chrome=False):
    """Lacquered or chrome cylinder: shading across, two specular stripes, sparkle flakes."""
    x = np.linspace(-1, 1, w, dtype=np.float32)
    base = np.clip(1 - np.abs(x + 0.15) ** 1.6, 0, 1)
    col = np.asarray(dark, np.float32) * (1 - base[:, None]) + np.asarray(light, np.float32) * base[:, None]
    img = np.broadcast_to(col[None], (h, w, 3)).copy()
    spec = np.exp(-((x + 0.45) / 0.07) ** 2) * 0.55 + np.exp(-((x - 0.55) / 0.05) ** 2) * 0.25
    if chrome:  # chrome reflects the room: dark and light bands
        bands = 0.35 * np.sin(x * 9.0 + 1.0) + 0.25 * np.sin(x * 23.0)
        img = img * (1 + bands[None, :, None] * 0.6)
        spec = spec * 1.4
    img = img + spec[None, :, None]
    rng = np.random.default_rng(seed)
    if not chrome:
        fl = (rng.random((h, w)) > 0.985).astype(np.float32) * rng.uniform(0.2, 0.6, (h, w)).astype(np.float32)
        fl = cv2.GaussianBlur(fl, (0, 0), 0.6) * 2.5
        img = img + fl[..., None] * (0.4 + base[None, :, None])
    return np.clip(img, 0, 1)


def _head_tone(h, w, cx, cy, rx, ry, seed=0):
    """Coated drum head: off-white, a little texture, stick wear in the middle, light from above."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    img = np.broadcast_to(np.array(HEAD, np.float32), (h, w, 3)).copy()
    img *= (0.93 + 0.07 * ((yy - cy) / ry * -0.5 + 0.5))[..., None]
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * 0.025
    img += n[..., None]
    wear = np.exp(-((rr - 0.12) / 0.22) ** 2) * 0.07
    img -= wear[..., None] * np.array([1.0, 1.0, 0.9], np.float32)
    img *= (1 - 0.10 * np.clip(rr - 0.75, 0, 1) / 0.25)[..., None]
    return np.clip(img, 0, 1)


def drum(rx, depth, shell=RED_SHELL, ink=4, lugs=6, tilt=0.30, chrome=False, seed=0):
    """A drum seen from the front and a little above, with chrome hoops, tension rods and lugs.
    Anchor 'head' = center of the top head."""
    ry = rx * tilt
    hoop = max(4.0, rx * 0.06)
    pad = ink + hoop + 6
    w, h = int(2 * rx + 2 * pad), int(2 * ry + depth + 2 * pad + hoop)
    img = _canvas(w, h)
    cx, cy = w / 2, pad + ry
    body = np.concatenate([_ellipse_pts(cx, cy, rx, ry, math.pi, 2 * math.pi, 60),
                           _ellipse_pts(cx, cy + depth, rx, ry, 0, math.pi, 60)])
    _poly(img, body, _sparkle_shell(h, w, *(CHROME if chrome else shell), seed=seed, chrome=chrome), ink=0)
    # shadow under the top hoop
    sh = np.zeros((h, w), np.float32)
    cv2.ellipse(sh, (int(cx), int(cy + hoop * 0.6)), (int(rx), int(ry)), 0, 0, 180, 1.0, int(hoop * 1.6), cv2.LINE_AA)
    sh = cv2.GaussianBlur(sh, (0, 0), hoop * 0.5) * (img[..., 3])
    img[..., :3] *= (1 - 0.45 * sh[..., None])
    # tension rods and lugs
    for i in range(lugs):
        a = math.pi * (0.10 + 0.80 * (i + 0.5) / lugs)
        x = cx - rx * math.cos(a)
        yb = cy + ry * math.sin(a)
        lug_y0, lug_y1 = yb + depth * 0.30, yb + depth * 0.62
        _line(img, [(x, yb + hoop), (x, lug_y0)], max(1.5, rx * 0.012), color=CHROME[1])
        _poly(img, _chaikin([(x - rx * 0.03, lug_y0), (x + rx * 0.03, lug_y0), (x + rx * 0.03, lug_y1), (x - rx * 0.03, lug_y1)], 2),
              _gradient(h, w, (0.55, 0.55, 0.58), (0.98, 0.98, 1.0)), ink=1.5)
        if chrome is False and depth > rx * 0.5:  # bottom rods on deep drums
            _line(img, [(x, lug_y1), (x, yb + depth - hoop)], max(1.5, rx * 0.012), color=CHROME[1])
    # bottom hoop
    for wv, col in ((hoop + 2, INK), (hoop, CHROME[1]), (hoop * 0.35, (1.0, 1.0, 1.0))):
        _line(img, _ellipse_pts(cx, cy + depth - hoop * 0.3, rx, ry, 0.05, math.pi - 0.05, 60), wv, color=col)
    # top head and hoop
    _poly(img, _ellipse_pts(cx, cy, rx - hoop * 0.5, ry - hoop * 0.3), _head_tone(h, w, cx, cy, rx, ry, seed), ink=0)
    for wv, col in ((hoop + 2.5, INK), (hoop, CHROME[0]), (hoop * 0.55, CHROME[1])):
        _line(img, _ellipse_pts(cx, cy, rx - hoop * 0.3, ry - hoop * 0.2), wv, closed=True, color=col)
    _line(img, _ellipse_pts(cx, cy - hoop * 0.25, rx - hoop * 0.3, ry - hoop * 0.2, math.pi * 1.15, math.pi * 1.6, 30), hoop * 0.3, color=(1, 1, 1))
    _line(img, body, ink, closed=False)
    img = marker(img, math.pi / 2, seed=seed)
    return img, dict(head=(cx, cy), top=(cx, cy))


def kick_front(r, skull=None, ink=5):
    """Bass drum seen from the front: lacquered wood hoop with metal claws, black resonant head
    with a port hole, Ben's skull on it."""
    pad = ink + 8
    w = h = int(2 * r + 2 * pad)
    img = _canvas(w, h)
    c = w / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.arctan2(yy - c, xx - c)
    hoop_col = _sparkle_shell(h, w, *RED_SHELL, seed=3) * (0.85 + 0.25 * np.cos(ang + 2.2))[..., None]
    _poly(img, _ellipse_pts(c, c, r, r, n=200), np.clip(hoop_col, 0, 1), ink=0)
    rr = np.hypot(xx - c, yy - c) / r
    headc = np.array([0.10, 0.09, 0.11], np.float32) * (1.0 + 0.6 * np.exp(-((xx - c + 0.35 * r) ** 2 + (yy - c + 0.4 * r) ** 2) / (0.5 * r) ** 2))[..., None]
    _poly(img, _ellipse_pts(c, c, r * 0.88, r * 0.88, n=200), np.clip(headc, 0, 1), ink=0)
    for wv, col in ((10, INK), (6, CHROME[0]), (3, CHROME[1])):
        _line(img, _ellipse_pts(c, c, r * 0.885, r * 0.885, n=200), wv * r / 260, closed=True, color=col)
    img = marker(img, 0.5, seed=11)
    if skull is not None:
        sk = skull.copy()
        a = cv2.erode(sk[..., 3], np.ones((5, 5), np.uint8))
        sk = sk.astype(np.float32) / 255
        sk[..., 3] = cv2.GaussianBlur(a.astype(np.float32) / 255, (0, 0), 1.2)
        s = (r * 1.05) / sk.shape[0]
        sk = cv2.resize(sk, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        sh_, sw = sk.shape[:2]
        y0, x0 = int(c - sh_ / 2 - r * 0.06), int(c - sw / 2)
        aa = sk[..., 3:4]
        img[y0:y0 + sh_, x0:x0 + sw, :3] = img[y0:y0 + sh_, x0:x0 + sw, :3] * (1 - aa) + sk[..., :3] * aa
    # port hole, lower right
    px, py, pr = c + r * 0.45, c + r * 0.50, r * 0.13
    _poly(img, _ellipse_pts(px, py, pr, pr, n=50), (0.03, 0.03, 0.035), ink=0)
    _line(img, _ellipse_pts(px, py, pr, pr, n=50), 3 * r / 260, closed=True, color=CHROME[1])
    for i in range(12):  # claws and tension rods on the hoop
        a = 2 * math.pi * i / 12 + 0.26
        x0_, y0_ = c + r * 0.86 * math.cos(a), c + r * 0.86 * math.sin(a)
        x1_, y1_ = c + r * 1.0 * math.cos(a), c + r * 1.0 * math.sin(a)
        _line(img, [(x0_, y0_), (x1_, y1_)], 7 * r / 260, color=INK)
        _line(img, [(x0_, y0_), (x1_, y1_)], 4 * r / 260, color=CHROME[1])
    hl = np.zeros((h, w), np.float32)
    cv2.ellipse(hl, (int(c), int(c)), (int(r * 0.95), int(r * 0.95)), 0, 200, 250, 1.0, max(2, int(r * 0.03)), cv2.LINE_AA)
    hl = cv2.GaussianBlur(hl, (0, 0), r * 0.01 + 1)
    img[..., :3] = img[..., :3] * (1 - 0.6 * hl[..., None]) + 0.6 * hl[..., None]
    _line(img, _ellipse_pts(c, c, r, r, n=200), ink, closed=True)
    return img, dict(center=(c, c))


def cymbal(rx, tilt=0.22, ink=3):
    """Brass cymbal with fine lathe grooves, a bell, a felt and wing nut on top."""
    ry = rx * tilt
    pad = ink + 6
    w, h = int(2 * rx + 2 * pad), int(2 * ry + 2 * pad + rx * 0.22)
    img = _canvas(w, h)
    cx, cy = w / 2, h / 2 + rx * 0.05
    out = np.concatenate([_ellipse_pts(cx, cy, rx, ry, 0, math.pi, 80), _ellipse_pts(cx, cy - rx * 0.025, rx, ry * 0.86, math.pi, 2 * math.pi, 80)])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    ang = np.arctan2((yy - cy) / ry, (xx - cx) / rx)
    grooves = 0.5 + 0.5 * np.sin(rr * 140)
    aniso = 0.5 + 0.5 * np.cos(2 * (ang + 0.6))  # anisotropic sheen of a lathed surface
    tone = 0.45 + 0.35 * aniso + 0.08 * grooves - 0.15 * rr
    col = np.asarray(BRASS[0], np.float32) * (1 - tone[..., None]) + np.asarray(BRASS[1], np.float32) * tone[..., None]
    _poly(img, out, np.clip(col, 0, 1), ink=0)
    _poly(img, _ellipse_pts(cx, cy - ry * 0.28, rx * 0.17, ry * 0.45), _gradient(h, w, BRASS[1], BRASS[0]), ink=1.5)  # bell
    _poly(img, [(cx - rx * 0.05, cy - ry * 0.75), (cx + rx * 0.05, cy - ry * 0.75), (cx + rx * 0.05, cy - ry * 0.55), (cx - rx * 0.05, cy - ry * 0.55)],
          (0.25, 0.25, 0.55), ink=1.5)  # felt
    _poly(img, _ellipse_pts(cx, cy - ry * 0.9, rx * 0.06, rx * 0.035, n=20), CHROME[1], ink=1.5)  # wing nut
    _line(img, _ellipse_pts(cx, cy, rx, ry, 0.1, math.pi - 0.1, 60), 2.0, color=BRASS[0])  # edge
    _line(img, out, ink, closed=True)
    img = marker(img, 0.12, seed=int(rx))
    return img, dict(center=(cx, cy))


def stand(dst, top, floor_y, width=6, legs=True, tilter=True):
    """Cymbal or drum stand drawn into the frame: tube, a joint clamp, tripod legs at the floor."""
    x, y = top
    pole(dst, (x, y), (x + 4, floor_y - 10), width)
    jy = y + (floor_y - y) * 0.45
    cv2.rectangle(dst, (int(x - width * 1.2), int(jy - width)), (int(x + width * 1.2 + 2), int(jy + width)), INK, -1, cv2.LINE_AA)
    cv2.rectangle(dst, (int(x - width), int(jy - width + 2)), (int(x + width + 2), int(jy + width - 2)), (0.7, 0.7, 0.74), -1, cv2.LINE_AA)
    if legs:
        for dx in (-1, 1):
            pole(dst, (x + 4, floor_y - 40), (x + 4 + dx * 70, floor_y), max(3, width - 2))
        pole(dst, (x + 4, floor_y - 40), (x + 14, floor_y + 4), max(3, width - 2))


def tom_mount(dst, a, b, width=7):
    """Chrome L-arm from the bass drum to a tom."""
    mid = (b[0], a[1] - (a[1] - b[1]) * 0.15)
    pole(dst, a, mid, width)
    pole(dst, mid, b, width)


def floor_shadow(dst, cx, y, rx, ry, k=0.5):
    """Soft contact shadow on the stage floor."""
    h, w = dst.shape[:2]
    m = np.zeros((h // 4, w // 4), np.float32)
    cv2.ellipse(m, (int(cx / 4), int(y / 4)), (max(1, int(rx / 4)), max(1, int(ry / 4))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 3), (w, h))[..., None]
    dst *= 1 - k * m


def stick(L=230, ink=3):
    w, h = 28, int(L + 16)
    img = _canvas(w, h)
    cx = w / 2
    body = [(cx - 6, h - 8), (cx + 6, h - 8), (cx + 3.5, 18), (cx - 3.5, 18)]
    _poly(img, body, _shell_tone(h, w, (0.30, 0.55, 0.72), (0.62, 0.85, 0.96)), ink=ink)
    _poly(img, _ellipse_pts(cx, 13, 5.5, 8, n=24), (0.62, 0.85, 0.96), ink=ink)
    # anchors: tip at the top, grip near the butt end
    return img, dict(tip=(cx, 12), grip=(cx, h - 8 - 0.18 * L))


def pole(dst, p0, p1, width=7):
    """A stand pole: dark metal with a highlight, drawn straight into the frame."""
    p0, p1 = tuple(int(v) for v in p0), tuple(int(v) for v in p1)
    cv2.line(dst, p0, p1, INK, width + 4, cv2.LINE_AA)
    cv2.line(dst, p0, p1, (0.35, 0.35, 0.38), width, cv2.LINE_AA)
    cv2.line(dst, (p0[0] - 1, p0[1]), (p1[0] - 1, p1[1]), (0.75, 0.75, 0.78), max(1, width // 3), cv2.LINE_AA)


# ---------------------------------------------------------------- electric guitar
def _chaikin(pts, n=3):
    P = np.asarray(pts, float)
    for _ in range(n):
        Q = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        P = np.array(Q)
    return P


def guitar(Lb, ink=5, neck=1.55, burst=((0.25, 0.55, 0.85), (0.22, 0.20, 0.10)), n_strings=6):
    """Single-cutaway electric guitar, neck pointing right (+x), strings drawn per frame.
    Lb = body length in px. Anchors: strum (over the pickups), bridge, nut, and the string
    spacing, so callers can draw ringing strings. A bass: neck=2.1, n_strings=4, darker burst."""
    neck_len, neck_w, head_len = neck * Lb, (0.13 if n_strings == 6 else 0.12) * Lb, 0.36 * Lb
    pad = ink + 6
    w = int(Lb + neck_len + head_len + 2 * pad)
    h = int(0.9 * Lb + 2 * pad)
    img = _canvas(w, h)
    ox, oy = pad, h / 2
    B = lambda pts: [(ox + x * Lb, oy + y * Lb) for x, y in pts]  # noqa: E731
    body = _chaikin(B([(0, 0), (0.03, -0.26), (0.15, -0.41), (0.31, -0.44), (0.48, -0.39), (0.61, -0.31), (0.72, -0.32),
                       (0.84, -0.34), (0.95, -0.27), (1.0, -0.12), (1.02, -0.06), (1.02, 0.06), (0.97, 0.09), (0.92, 0.10),
                       (0.87, 0.17), (0.86, 0.25), (0.81, 0.30), (0.72, 0.31), (0.61, 0.31), (0.48, 0.39), (0.31, 0.44),
                       (0.15, 0.41), (0.03, 0.26)]), 3)
    # sunburst: dark teal edge to a warm amber center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt(((xx - ox - 0.42 * Lb) / (0.62 * Lb)) ** 2 + ((yy - oy) / (0.42 * Lb)) ** 2)
    k = np.clip((rr - 0.25) / 0.75, 0, 1)[..., None] ** 1.3
    burst = np.array(burst[0], np.float32) * (1 - k) + np.array(burst[1], np.float32) * k
    _poly(img, body, burst, ink=0)
    # neck and headstock
    nx0, nx1 = ox + 0.92 * Lb, ox + Lb + neck_len
    _poly(img, [(nx0, oy - neck_w / 2), (nx1, oy - neck_w * 0.45), (nx1, oy + neck_w * 0.45), (nx0, oy + neck_w / 2)],
          _gradient(h, w, (0.35, 0.55, 0.75), (0.22, 0.38, 0.55), axis=0), ink=ink - 1)
    fb0 = ox + 1.0 * Lb
    for i in range(1, 19):  # frets get closer toward the body
        x = nx1 - neck_len * (1 - 2 ** (-i / 12)) * 1.45
        if x > fb0:
            _line(img, [(x, oy - neck_w * 0.46), (x, oy + neck_w * 0.46)], 2, color=(0.75, 0.78, 0.80))
    for i in (3, 5, 7, 9, 12, 15):
        x = nx1 - neck_len * (1 - 2 ** (-(i - 0.5) / 12)) * 1.45
        if x > fb0:
            _poly(img, _ellipse_pts(x, oy, 3.5, 3.5, n=12), (0.92, 0.92, 0.90), ink=0)
    head = [(nx1 - 2, oy - neck_w * 0.5), (nx1 + head_len * 0.9, oy - neck_w * 0.95), (nx1 + head_len, oy - neck_w * 0.7),
            (nx1 + head_len, oy + neck_w * 0.6), (nx1 + head_len * 0.85, oy + neck_w * 0.75), (nx1 - 2, oy + neck_w * 0.5)]
    _poly(img, head, (0.10, 0.10, 0.12), ink=ink - 1)
    for i in range(3):
        x = nx1 + head_len * (0.25 + 0.25 * i)
        for sgn in (-1, 1):
            _poly(img, _ellipse_pts(x, oy + sgn * neck_w * 0.95, 6, 5, n=14), (0.85, 0.85, 0.88), ink=2)
    # pickguard, pickups, bridge, knobs
    guard = _chaikin(B([(0.45, 0.06), (0.62, 0.08), (0.82, 0.10), (0.84, 0.22), (0.70, 0.27), (0.52, 0.27), (0.42, 0.20)]), 2)
    _poly(img, guard, (0.80, 0.90, 0.95), ink=2)
    for x in (0.58, 0.80):
        _poly(img, [(ox + (x - 0.05) * Lb, oy - 0.10 * Lb), (ox + (x + 0.05) * Lb, oy - 0.10 * Lb),
                    (ox + (x + 0.05) * Lb, oy + 0.10 * Lb), (ox + (x - 0.05) * Lb, oy + 0.10 * Lb)], (0.12, 0.12, 0.13), ink=2)
        for j in range(-2, 3):
            _poly(img, _ellipse_pts(ox + x * Lb, oy + j * 0.035 * Lb, 2.5, 2.5, n=10), (0.75, 0.75, 0.78), ink=0)
    bx = ox + 0.33 * Lb
    _poly(img, [(bx - 0.03 * Lb, oy - 0.11 * Lb), (bx + 0.03 * Lb, oy - 0.11 * Lb), (bx + 0.03 * Lb, oy + 0.11 * Lb), (bx - 0.03 * Lb, oy + 0.11 * Lb)],
          _gradient(h, w, *CHROME, axis=0), ink=2)
    for (x, y) in ((0.22, 0.20), (0.30, 0.28), (0.14, 0.28)):
        _poly(img, _ellipse_pts(ox + x * Lb, oy + y * Lb, 0.035 * Lb, 0.035 * Lb, n=20), (0.25, 0.70, 0.85), ink=2)
    hl = np.zeros((h, w), np.float32)
    cv2.ellipse(hl, (int(ox + 0.25 * Lb), int(oy - 0.25 * Lb)), (int(0.16 * Lb), int(0.05 * Lb)), -20, 0, 360, 1.0, -1, cv2.LINE_AA)
    hl = cv2.GaussianBlur(hl, (0, 0), 5)
    img[..., :3] = img[..., :3] * (1 - 0.45 * hl[..., None]) + 0.45 * hl[..., None]
    _line(img, body, ink, closed=True)
    return img, dict(strum=(ox + 0.62 * Lb, oy), bridge=(bx, oy), nut=(nx1, oy), spacing=(0.022 if n_strings == 6 else 0.03) * Lb, neck_w=neck_w,
                     neck=(ox + Lb, oy), neck_len=neck_len, n=n_strings)


def strings(img, anc, ring, t, color=(0.85, 0.86, 0.88)):
    """Six strings from bridge to nut; ringing strings blur into a wobbly band."""
    out = img.copy()
    (bx, by), (nx, ny) = anc["bridge"], anc["nut"]
    sp = anc["spacing"]
    n = anc.get("n", 6)
    xs = np.linspace(bx, nx, 60)
    env = np.sin(np.pi * (xs - bx) / (nx - bx))
    for i in range(n):
        y0 = by + (i - (n - 1) / 2) * sp
        y1 = ny + (i - (n - 1) / 2) * sp * 0.75
        ys = y0 + (y1 - y0) * (xs - bx) / (nx - bx)
        amp = ring * (2.5 + 0.4 * i)
        for k, ph in enumerate((0.0, math.pi)):
            yk = ys + amp * env * math.sin(2 * math.pi * (55 + 9 * i) * t + ph)
            _line(out, np.stack([xs, yk], 1), (1.4 if k == 0 else 1.0) * (1.6 if n == 4 else 1.0), color=color)
    return out


def strap(dst, pts, width, color=(0.10, 0.15, 0.24)):
    """A leather guitar strap along a curve: ink edge, leather, a lighter top edge, stitching."""
    P = _chaikin(np.asarray(pts, float), 0) if len(pts) > 2 else np.asarray(pts, float)
    # resample a smooth curve through the points (quadratic Bezier for 3 points)
    if len(pts) == 3:
        t = np.linspace(0, 1, 40)[:, None]
        a, b, c = (np.asarray(p, float) for p in pts)
        P = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c
    Q = np.round(P * 4).astype(np.int32)
    cv2.polylines(dst, [Q], False, INK, int(width + 6), cv2.LINE_AA, shift=2)
    cv2.polylines(dst, [Q], False, color, int(width), cv2.LINE_AA, shift=2)
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    for side, col in ((-0.30, (0.30, 0.40, 0.55)), (0.30, (0.20, 0.28, 0.40))):
        E = P + n * side * width
        for k in range(0, len(E) - 1, 2):  # stitches: short dashes
            cv2.line(dst, tuple(np.round(E[k] * 4).astype(int)), tuple(np.round(E[k + 1] * 4).astype(int)), col, max(1, int(width * 0.08)), cv2.LINE_AA, shift=2)
    hl = P + n * (-0.42) * width
    cv2.polylines(dst, [np.round(hl * 4).astype(np.int32)], False, (0.35, 0.45, 0.6), max(1, int(width * 0.06)), cv2.LINE_AA, shift=2)


def button(dst, p, r):
    cv2.circle(dst, (int(p[0]), int(p[1])), int(r + 2), INK, -1, cv2.LINE_AA)
    cv2.circle(dst, (int(p[0]), int(p[1])), int(r), (0.85, 0.85, 0.88), -1, cv2.LINE_AA)
    cv2.circle(dst, (int(p[0] - r * 0.3), int(p[1] - r * 0.3)), max(1, int(r * 0.35)), (1, 1, 1), -1, cv2.LINE_AA)


def sleeve(dst, a, b, ra, rb, color, fur=(0.86, 0.88, 0.90), ink=3):
    """A painted-looking forearm sleeve from a (elbow / painted cuff) to b (wrist): tapered,
    shaded across, ink edge, a fur cuff at the wrist end."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = b - a
    L = np.hypot(*v)
    if L < 1:
        return
    u = v / L
    n = np.array([-u[1], u[0]])
    R = max(ra, rb) + ink + 6
    x0, y0 = np.floor(np.minimum(a, b) - R).astype(int)
    x1, y1 = np.ceil(np.maximum(a, b) + R).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, dst.shape[1]), min(y1, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    t = np.clip(((xx - a[0]) * u[0] + (yy - a[1]) * u[1]) / L, 0, 1)
    r = ra + (rb - ra) * t
    cx, cy = a[0] + v[0] * t, a[1] + v[1] * t
    across = ((xx - cx) * n[0] + (yy - cy) * n[1]) / np.maximum(r, 1)
    dist = np.hypot(xx - cx, yy - cy)
    m = np.clip(r - dist + 0.5, 0, 1)
    edge = np.clip(r + ink - dist + 0.5, 0, 1) - m
    lit = np.clip(0.6 - 0.6 * across, 0, 1)
    col = np.asarray(color, np.float32) * (0.62 + 0.45 * lit)[..., None]
    folds = 0.06 * np.sin(t * L / 9.0 + across * 2.0)
    col = col * (1 + folds)[..., None]
    # fur cuff: the last part near b, with a ragged edge
    cuff = (t > 1 - min(0.35, 30 / L)) & (np.sin(across * 9 + t * 40) * 0.04 + t > 1 - min(0.33, 28 / L))
    col = np.where(cuff[..., None], np.asarray(fur, np.float32) * (0.75 + 0.3 * lit)[..., None], col)
    reg = dst[y0:y1, x0:x1]
    reg[:] = reg * (1 - edge[..., None]) + np.asarray(INK, np.float32) * edge[..., None]
    reg[:] = reg * (1 - m[..., None]) + np.clip(col, 0, 1) * m[..., None]


# ---------------------------------------------------------------- marker look (like Ben's drawings)
def _streaks(h, w, angle, seed, length=40, width=1.6):
    """Marker-stroke texture: noise smeared along one direction (overlapping marker passes)."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.2)
    L = int(length)
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2
    dx, dy = math.cos(angle), math.sin(angle)
    for t in np.linspace(-c, c, L * 2):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < L and 0 <= y < L:
            k[y, x] = 1
    k = cv2.GaussianBlur(k, (0, 0), width * 0.5)
    k /= k.sum()
    s = cv2.filter2D(n, -1, k)
    return s / (s.std() + 1e-6)


def marker(img, angle=0.0, seed=0, tones=6, streak=0.07, ink_boost=1.6, hl=True):
    """Re-render a code-drawn BGRA sprite in a marker style: shading quantized to a few flat tones
    with soft steps, marker streaks along `angle`, the dark outline thickened and wobbly, and a few
    white highlight strokes on the light tones (as Ben does on hair and buttons)."""
    out = img.copy()
    a = img[..., 3]
    h, w = a.shape
    bgr = img[..., :3]
    lum = bgr.mean(2)
    ink = (lum < 0.12) & (a > 0.5)
    # flat tones: quantize lightness, keep hue
    q = np.floor(lum * tones + 0.5) / tones
    q = 0.65 * cv2.GaussianBlur(q.astype(np.float32), (0, 0), 1.5) + 0.35 * lum  # soft marker steps
    ratio = (q + 0.04) / (lum + 0.04)
    col = np.clip(bgr * ratio[..., None], 0, 1)
    # marker passes: streaks, stronger in mid tones; the paper shows a bit through light tones
    st = _streaks(h, w, angle, seed)
    col = col * (1 + streak * st[..., None] * (0.4 + 0.6 * (1 - np.abs(lum - 0.5) * 2))[..., None])
    # wobbly, thicker outline: dilate the ink a little with noise
    wob = cv2.GaussianBlur(np.random.default_rng(seed + 1).normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6)
    thick = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) & (wob > -0.3)
    edge = cv2.morphologyEx((a > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    inkm = np.clip(ink.astype(np.float32) + thick * 0.9 + edge * 0.9 * (ink_boost > 1), 0, 1)
    inkm = cv2.GaussianBlur(inkm, (0, 0), 0.6)
    col = col * (1 - inkm[..., None]) + np.array(INK, np.float32) * inkm[..., None]
    if hl:  # white highlight dashes on the brightest tone
        rng = np.random.default_rng(seed + 2)
        bright = (q > 0.72) & (a > 0.9) & ~ink
        ys, xs = np.nonzero(bright)
        if len(ys):
            hm = np.zeros((h, w), np.float32)
            for i in rng.choice(len(ys), size=min(len(ys), max(2, len(ys) // 2500)), replace=False):
                L = rng.uniform(10, 26)
                x0, y0 = xs[i], ys[i]
                cv2.line(hm, (int(x0), int(y0)), (int(x0 + L * math.cos(angle)), int(y0 + L * math.sin(angle))), 1.0, max(1, int(rng.uniform(1.5, 3))), cv2.LINE_AA)
            hm *= bright
            col = col * (1 - 0.8 * hm[..., None]) + 0.97 * 0.8 * hm[..., None]
    out[..., :3] = np.clip(col, 0, 1)
    out[..., 3] = np.maximum(a, cv2.dilate(a, np.ones((3, 3), np.uint8)) * 0.0)
    return out


def mic_stand(dst, x, top, floor, s):
    """A straight mic stand: round heavy base on the floor, tube with a clutch, a clip at the top."""
    w = max(4, int(16 * s))
    pole(dst, (x, top + 20 * s), (x, floor - 20 * s), w)
    cy = top + (floor - top) * 0.45
    cv2.rectangle(dst, (int(x - w * 1.3), int(cy - w * 1.4)), (int(x + w * 1.3), int(cy + w * 1.4)), INK, -1, cv2.LINE_AA)
    cv2.rectangle(dst, (int(x - w), int(cy - w)), (int(x + w), int(cy + w)), (0.55, 0.55, 0.58), -1, cv2.LINE_AA)
    cv2.ellipse(dst, (int(x), int(floor - 8 * s)), (int(150 * s), int(30 * s)), 0, 0, 360, INK, -1, cv2.LINE_AA)
    cv2.ellipse(dst, (int(x), int(floor - 14 * s)), (int(140 * s), int(22 * s)), 0, 0, 360, (0.22, 0.22, 0.24), -1, cv2.LINE_AA)
    cv2.ellipse(dst, (int(x - 40 * s), int(floor - 20 * s)), (int(60 * s), int(6 * s)), 0, 0, 360, (0.45, 0.45, 0.48), -1, cv2.LINE_AA)
    cv2.rectangle(dst, (int(x - w * 1.1), int(top)), (int(x + w * 1.1), int(top + 40 * s)), INK, -1, cv2.LINE_AA)  # clip
