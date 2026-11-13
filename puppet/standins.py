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
SRC = f"{ROOT}/public/chars/distor-front-v2.png"
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


def stroke(img, pts, width, closed=False, seed=0, color=None):
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
    col = INK if color is None else np.array(color, np.float32) / 255
    img[..., :3] = img[..., :3] * (1 - m[..., None]) + col * m[..., None]
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
    im = cv2.imread(SRC, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    cx, cy, r = 280, 915, 24
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


JACKET = (14, 10, 36)  # BGR, Ben's dark maroon jacket
TRIM = (152, 145, 140)  # the grey band on his collar, used for the cuffs
SHEEN = (52, 40, 112)  # lighter maroon for folds and the lit edge (black ink would vanish on the jacket)
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
    stroke(img, [(70, 250), (62, 450), (75, 650)], 14, seed=5, color=SHEEN)  # lit edge
    stroke(img, out, 12, closed=True, seed=1)
    stroke(img, arc((165, 205), 120, 55, math.radians(200), math.radians(340)), 8, seed=2, color=SHEEN)  # shoulder seam
    stroke(img, [(110, 640), (150, 660), (185, 655)], 8, seed=3, color=SHEEN)  # folds near the elbow
    stroke(img, [(200, 690), (235, 700)], 7, seed=4, color=SHEEN)
    return img, dict(shoulder=S, elbow=E)


def forearm_hand():
    w, h = 330, 900
    E, Wr = (165, 125), (165, 125 + 470)
    img = np.zeros((h, w, 4), np.float32)
    fy = Wr[1]
    # sleeve and cuff
    sleeve = capsule(E, 106, (165, fy - 100), 100, bulge=-4)
    m2 = fill(img, sleeve, tex("jacket", h, w, JACKET, 3), 13)
    stroke(img, [(78, 200), (72, 330), (80, fy - 110)], 13, seed=6, color=SHEEN)
    stroke(img, sleeve, 12, closed=True, seed=14)
    cuff = np.array([(62, fy - 95), (268, fy - 95), (272, fy + 2), (58, fy + 2)], float)
    cuff = np.concatenate([[(62 + (268 - 62) * t, fy - 95 + 6 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 20)],
                           [(270, fy - 40)], [(272 - 214 * t, fy + 2 + 8 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 20)], [(60, fy - 40)]])
    fill(img, cuff, tex("jacket", h, w, TRIM, 4), 15)
    stroke(img, cuff, 11, closed=True, seed=16)
    paste(img, button_sprite(), 225, fy - 46)
    stroke(img, [(120, 215), (160, 240)], 7, seed=17, color=SHEEN)  # sleeve folds near the elbow
    stroke(img, [(190, 230), (215, 262)], 7, seed=18, color=SHEEN)
    return img, dict(elbow=E, wrist=(165, fy - 8))  # the hand is a rigged hand (puppet/hand.py)


EXT = 450  # rows added below the page edge


def body():
    """Pad the drawing, continue the shoulders and jacket sides past the page edge, and
    continue the jacket below the page bottom (letters on the shirt fade into the black)."""
    im = cv2.imread(SRC, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
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
    # below the page: each column continues with the median color of its last rows (smoothed
    # sideways, so no streaks), blended in over the page edge, with jacket grain on top
    col = np.median(im[h - 60:h], axis=0)
    col[..., :3] = cv2.GaussianBlur(col[None, :, :3], (0, 0), sigmaX=9, sigmaY=0)[0]
    noise = tex("jacket", EXT + 40, w, (128, 128, 128), 7) - 0.5
    blk = np.repeat(col[None], EXT + 40, 0)
    blk[..., :3] += noise * 0.5 * blk[..., 3:4]
    k = np.clip(np.arange(40, dtype=np.float32) / 40, 0, 1)[:, None, None]
    big[h - 40:h, PAD:PAD + w] = big[h - 40:h, PAD:PAD + w] * (1 - k) + blk[:40] * k
    big[h:, PAD:PAD + w] = blk[40:]
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


def cut_v2():
    """Ben's new singer page (dark jacket, light blue paper): remove the paper and the signature,
    crop so it lines up with the first drawing (same head, 20 px wider page)."""
    im = cv2.imread(f"{ROOT}/drawings/ben/distor-front-v2.png")
    paper = np.array([243, 237, 224], np.float32)
    d = np.sqrt(((im.astype(np.float32) - paper) ** 2).sum(2))
    bg = (d < 40).astype(np.uint8) * 255
    mask = np.zeros((im.shape[0] + 2, im.shape[1] + 2), np.uint8)
    for x in range(0, im.shape[1], 7):
        if bg[0, x] == 255:
            cv2.floodFill(bg, mask, (x, 0), 128)
    fg = (bg != 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.float32)
    a = cv2.GaussianBlur(fg, (0, 0), 0.7)
    out = np.dstack([im, (a * 255).astype(np.uint8)])[:, 20:1170]
    cv2.imwrite(SRC, out)


def save(path, img):
    cv2.imwrite(path, np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))


def main():
    os.makedirs(OUT, exist_ok=True)
    cut_v2()
    ua, ua_j = upper_arm()
    fa, fa_j = forearm_hand()
    save(f"{OUT}/upper_arm.png", ua)
    save(f"{OUT}/forearm.png", fa)
    save(f"{OUT}/body.png", body())
    j = dict(credit="Drawing: Ben (arms and jacket sides: code stand-ins)",
             pad=PAD, hand=dict(skin_bgr=list(SKIN), length=360), upper_arm=dict(file="upper_arm.png", **{k: list(v) for k, v in ua_j.items()}),
             forearm=dict(file="forearm.png", **{k: list(v) for k, v in fa_j.items()}),
             shoulders=dict(left=[PAD - 85, 1455], right=[PAD + 1150 + 90, 1430]), arm_width=230)
    json.dump(j, open(f"{OUT}/parts.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
