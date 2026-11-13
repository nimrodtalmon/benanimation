"""Stand-in parts until Ben draws them, in the same format he will deliver
(one image per part + joint points), so his drawings can replace them 1:1.

Distor (Ben's singer):
  - body.png: his front drawing, padded, with the shoulders and jacket sides continued past
    the page edge (the drawing stops at the paper border).
  - upper_arm.png, forearm.png: code-drawn in his marker style (black outline, his own
    jacket/skin textures, a button copied from his drawing).
  python3 puppet/standins.py  -> public/parts/distor/
"""
import json
import math
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/public/parts/distor"
INK = np.array([19, 18, 20], np.float32) / 255  # BGR, Ben's marker black
PAD = 260  # body canvas padding left/right


def tex(name, h, w, mean_bgr=None, seed=0):
    t = cv2.imread(f"{ROOT}/public/tex/{name}.png").astype(np.float32) / 255
    reps = (h // t.shape[0] + 2, w // t.shape[1] + 2)
    big = np.tile(np.concatenate([t, t[:, ::-1]], 1), (reps[0], reps[1] // 2 + 1, 1))
    rng = np.random.default_rng(seed)
    oy, ox = rng.integers(0, t.shape[0]), rng.integers(0, t.shape[1])
    big = big[oy:oy + h, ox:ox + w]
    if mean_bgr is not None:
        big = big - big.reshape(-1, 3).mean(0) + np.array(mean_bgr, np.float32) / 255
    return np.clip(big, 0, 1)


def wobble(pts, amp, seed):
    """Marker lines are not perfect: displace points by a smooth random offset."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * math.pi, 4)
    i = np.arange(n)
    off = np.stack([amp * (np.sin(i * 0.11 + ph[0]) + 0.5 * np.sin(i * 0.37 + ph[1])),
                    amp * (np.sin(i * 0.13 + ph[2]) + 0.5 * np.sin(i * 0.29 + ph[3]))], 1)
    return pts + off


def stroke(img, pts, width, closed=False, seed=0):
    """Ink line with slightly varying width, antialiased, drawn into BGRA float img."""
    pts = wobble(pts, 1.2, seed)
    h, w = img.shape[:2]
    m = np.zeros((h * 1, w * 1), np.float32)
    P = np.round(pts * 8).astype(np.int32)
    seg = len(P) - (0 if closed else 1)
    for k in range(seg):
        a, b = P[k], P[(k + 1) % len(P)]
        ww = width * (0.85 + 0.3 * (0.5 + 0.5 * math.sin(k * 0.21 + seed)))
        cv2.line(m, tuple(a), tuple(b), 1.0, max(1, int(round(ww))), cv2.LINE_AA, shift=3)
    m = cv2.GaussianBlur(m, (0, 0), 0.7)
    img[..., :3] = img[..., :3] * (1 - m[..., None]) + INK * m[..., None]
    img[..., 3] = np.maximum(img[..., 3], m)


def fill(img, pts, color_img, seed=0):
    h, w = img.shape[:2]
    m = np.zeros((h * 8, w * 8), np.uint8)
    cv2.fillPoly(m, [np.round(wobble(pts, 1.0, seed) * 8).astype(np.int32)], 255, cv2.LINE_AA)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    img[..., :3] = img[..., :3] * (1 - m[..., None]) + color_img * m[..., None]
    img[..., 3] = np.maximum(img[..., 3], m)
    return m


def capsule(c0, r0, c1, r1, n=90, bulge=0.0):
    """Outline of a tapered capsule from circle (c0, r0) to (c1, r1), as a closed polygon."""
    c0, c1 = np.array(c0, float), np.array(c1, float)
    d = c1 - c0
    L = np.hypot(*d)
    u = d / L
    v = np.array([-u[1], u[0]])
    pts = []
    for t in np.linspace(0, 1, n):  # one side
        r = r0 + (r1 - r0) * t + bulge * math.sin(math.pi * t)
        pts.append(c0 + u * L * t + v * r)
    for a in np.linspace(0, math.pi, n // 2):  # bottom cap
        pts.append(c1 + v * r1 * math.cos(a) + u * r1 * math.sin(a))
    for t in np.linspace(1, 0, n):
        r = r0 + (r1 - r0) * t + bulge * math.sin(math.pi * t)
        pts.append(c0 + u * L * t - v * r)
    for a in np.linspace(math.pi, 2 * math.pi, n // 2):  # top cap
        pts.append(c0 + v * r0 * math.cos(a) + u * r0 * math.sin(a))
    return np.array(pts)


def arc(c, rx, ry, a0, a1, n=40):
    return np.array([(c[0] + rx * math.cos(a), c[1] + ry * math.sin(a)) for a in np.linspace(a0, a1, n)])


def button_sprite():
    """Copy one gold button (with its ink ring) from Ben's drawing."""
    im = cv2.imread(f"{ROOT}/public/chars/distor-front.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    cx, cy, r = 275, 917, 27
    patch = im[cy - r:cy + r, cx - r:cx + r].copy()
    yy, xx = np.mgrid[-r:r, -r:r]
    patch[..., 3] *= np.clip((r - 2 - np.hypot(xx, yy)) / 3, 0, 1)
    return patch


def paste(img, sprite, cx, cy):
    h, w = sprite.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    a = sprite[..., 3:4]
    img[y0:y0 + h, x0:x0 + w, :3] = img[y0:y0 + h, x0:x0 + w, :3] * (1 - a) + sprite[..., :3] * a
    img[y0:y0 + h, x0:x0 + w, 3:4] = np.maximum(img[y0:y0 + h, x0:x0 + w, 3:4], a)


JACKET = (172, 180, 180)  # BGR, sampled near the page edge of his jacket
TRIM = (205, 190, 182)
SKIN = (167, 199, 225)


def shade(img, mask, side_dir, amount, seed=0):
    """A flat marker shadow tone on one side of a part (Ben uses a few flat tones)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (xx - w / 2) * side_dir / (w / 2)
    k = np.clip((g - 0.35) / 0.12, 0, 1) * mask * amount
    img[..., :3] *= (1 - k[..., None])


def upper_arm():
    w, h = 330, 860
    S, E = (165, 165), (165, 165 + 600)
    img = np.zeros((h, w, 4), np.float32)
    out = capsule(S, 128, E, 106, bulge=6)
    m = fill(img, out, tex("jacket", h, w, JACKET, 1), 1)
    shade(img, m, -1, 0.10)
    stroke(img, out, 12, closed=True, seed=1)
    stroke(img, arc((165, 205), 120, 55, math.radians(200), math.radians(340)), 9, seed=2)  # shoulder seam
    stroke(img, [(110, 640), (150, 660), (185, 655)], 8, seed=3)  # folds near the elbow
    stroke(img, [(200, 690), (235, 700)], 7, seed=4)
    return img, dict(shoulder=S, elbow=E)


def forearm_hand():
    w, h = 330, 900
    E, Wr = (165, 125), (165, 125 + 470)
    img = np.zeros((h, w, 4), np.float32)
    # fist under the cuff: fingers wrapped toward the viewer, thumb on top
    fy = Wr[1]
    fist = np.concatenate([
        arc((165, fy + 10), 92, 40, math.radians(180), math.radians(360), 20),
        arc((165 + 70, fy + 95), 34, 95, math.radians(-80), math.radians(80), 20),
        arc((165, fy + 175), 88, 38, math.radians(10), math.radians(170), 24),
        arc((165 - 75, fy + 95), 30, 90, math.radians(100), math.radians(260), 20)])
    m = fill(img, fist, tex("skin", h, w, SKIN, 2), 5)
    shade(img, m, 1, 0.12)
    stroke(img, fist, 11, closed=True, seed=6)
    # fingers wrap across the grip: three gaps across the fist, thumb along the outer side
    for k, y in enumerate((fy + 72, fy + 110, fy + 148)):
        stroke(img, [(88 + 3 * k, y - 4), (130, y + 2), (170, y + 3), (196, y)], 7, seed=7 + k)
    stroke(img, [(222, fy + 30), (236, fy + 80), (232, fy + 130)], 8, seed=11)  # thumb
    # sleeve and cuff
    sleeve = capsule(E, 106, (165, fy - 40), 100, bulge=-4)
    m2 = fill(img, sleeve, tex("jacket", h, w, JACKET, 3), 13)
    shade(img, m2, -1, 0.10)
    stroke(img, sleeve, 12, closed=True, seed=14)
    cuff = np.array([(62, fy - 95), (268, fy - 95), (272, fy + 2), (58, fy + 2)], float)
    cuff = np.concatenate([[(62 + (268 - 62) * t, fy - 95 + 6 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 20)],
                           [(270, fy - 40)], [(272 - 214 * t, fy + 2 + 8 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 20)], [(60, fy - 40)]])
    fill(img, cuff, tex("jacket", h, w, TRIM, 4), 15)
    stroke(img, cuff, 11, closed=True, seed=16)
    paste(img, button_sprite(), 225, fy - 46)
    stroke(img, [(120, 215), (160, 240)], 7, seed=17)  # sleeve folds near the elbow
    stroke(img, [(190, 230), (215, 262)], 7, seed=18)
    return img, dict(elbow=E, wrist=Wr, grip=(165, fy + 95))


EXT = 450  # rows added below the page edge


def body():
    """Pad the drawing, continue the shoulders and jacket sides past the page edge, and
    continue the jacket below the page bottom (letters on the shirt fade into the black)."""
    im = cv2.imread(f"{ROOT}/public/chars/distor-front.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    h, w = im.shape[:2]
    # the paper edge: let the drawing fade out over the last pixels so the continuation blends in
    edge = np.clip(np.minimum(np.arange(w), w - 1 - np.arange(w)) / 14.0, 0, 1).astype(np.float32)
    im[1290:, :, 3] *= edge[None, :]
    # fade the red print at the bottom edge into the shirt black, so it does not end in a cut
    r = im[..., 2] - np.maximum(im[..., 0], im[..., 1])
    red = np.clip((r - 0.06) / 0.1, 0, 1)
    k = np.clip((np.arange(h, dtype=np.float32) - (h - 130)) / 90, 0, 1)[:, None] * red
    im[..., :3] = im[..., :3] * (1 - k[..., None]) + np.array([0.06, 0.06, 0.07], np.float32) * k[..., None]
    H = h + EXT
    big = np.zeros((H, w + 2 * PAD, 4), np.float32)
    big[:h, PAD:PAD + w] = im
    # below the page: repeat the last rows, with jacket texture so it does not look streaked
    last = im[h - 8:h].mean(0)
    noise = tex("jacket", EXT, w, (128, 128, 128), 7) - 0.5
    big[h:, PAD:PAD + w] = last[None]
    big[h:, PAD:PAD + w, :3] += noise * 0.6 * big[h:, PAD:PAD + w, 3:4]
    ext = np.zeros_like(big)
    X = lambda x: x + PAD  # noqa: E731
    # sides: shoulder slope continues from the page edge, then the jacket side goes down
    left = [(X(4), 1300), (X(-60), 1350), (X(-150), 1400), (X(-175), 1480), (X(-182), 1700), (X(-178), H), (X(4), H)]
    right = [(X(w - 4), 1282), (X(w + 60), 1318), (X(w + 150), 1372), (X(w + 178), 1460), (X(w + 184), 1700), (X(w + 180), H), (X(w - 4), H)]
    for poly, seed, xs in ((left, 20, slice(4, 40)), (right, 21, slice(w - 40, w - 4))):
        edge = np.median(im[1350:1590, xs, :3].reshape(-1, 3), axis=0) * 255
        fill(ext, np.array(poly, float), tex("jacket", H, w + 2 * PAD, edge, seed), seed)
        stroke(ext, np.array(poly[:6], float), 12, seed=seed + 10)
    a = big[..., 3:4]
    out = ext.copy()
    out[..., :3] = ext[..., :3] * (1 - a) + big[..., :3] * a
    out[..., 3:4] = ext[..., 3:4] * (1 - a) + a
    return np.clip(out, 0, 1)


def save(path, img):
    cv2.imwrite(path, np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))


def main():
    os.makedirs(OUT, exist_ok=True)
    ua, ua_j = upper_arm()
    fa, fa_j = forearm_hand()
    save(f"{OUT}/upper_arm.png", ua)
    save(f"{OUT}/forearm.png", fa)
    save(f"{OUT}/body.png", body())
    j = dict(credit="Drawing: Ben (arms and jacket sides: code stand-ins)",
             pad=PAD, upper_arm=dict(file="upper_arm.png", **{k: list(v) for k, v in ua_j.items()}),
             forearm=dict(file="forearm.png", **{k: list(v) for k, v in fa_j.items()}),
             shoulders=dict(left=[PAD - 85, 1455], right=[PAD + 1150 + 90, 1430]), arm_width=230)
    json.dump(j, open(f"{OUT}/parts.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
