"""A bust drawing (head + chest, the figure running off the page at the sides and bottom) ->
a puppet body: the paper removed, the jacket continued past the page edges (down to a hem, a little
out at the sides), plus code stand-in arms in the drawing's marker style. Same output format as
puppet/standins.py (body.png, upper_arm.png, forearm.png, parts.json), so Ben's own drawings of the
missing parts can replace the stand-ins 1:1.

  python3 puppet/build_bust.py blonde      -> public/parts/blonde/ (from public/rigs/blonde.json)

Each character is one settings file, public/rigs/<name>.json (coordinates on the light-flattened
photo): src (the photo), credit, how to cut it (seeds/right_edge/bottom, or kind "hoodie" with crop,
head_box and the hoodie outline), the clothes' colors, shoulders, arm and hand sizes, the head
landmarks for the puppet (head), and how the character plays (play: instrument, strum point,
neck angle, guitar size, strap ends; in body pixels). Optional: dark_eyes (solid black eye shapes:
glare removed), shirt/shirt_top (glare on a black shirt), blink (false for drawn eye sockets).
"""
import json
import math
import os
import sys

import cv2
import numpy as np

import photo
from standins import arc, capsule, fill, stroke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_page(c):
    im = cv2.imread(f"{ROOT}/{c['src']}")
    im = photo.flat_light(im)
    h, w = im.shape[:2]
    seeds = [(x, 5) for x in range(0, w, 15)]
    lim = int(c["seeds"].split(":")[1])
    seeds += [(5, y) for y in range(0, lim, 15)]
    a_, b_ = c["right_edge"]
    seeds += [(int(min(w - 6, a_ + b_ * y - 12)), y) for y in range(0, lim, 15)]
    alpha = photo.cut_figure(im, seeds, tol=28)
    # beyond the paper: nothing
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    alpha *= np.clip((a_ + b_ * Y - 6 - X) / 3.0, 0, 1)
    alpha *= np.clip((c["bottom"] - 4 - Y) / 3.0, 0, 1)
    return im, alpha


def jacket_tex(h, w, base, seed):
    """Marker-white with a faint paper grain."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2)
    n2 = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 12)
    t = np.array(base, np.float32)[None, None] / 255 + (0.012 * n + 0.02 * n2)[..., None]
    return np.clip(t, 0, 1)


def crayon(img, mask, side_dir, color, width, amount, seed):
    """Ben's crayon shading: an orange-brown band inside one edge of a shape, with stroke grain."""
    m = (mask > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    h, w = m.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    # only the edge facing side_dir (a unit vector): sample the distance field's gradient
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
    facing = np.clip(-(gx * side_dir[0] + gy * side_dir[1]) / (np.hypot(gx, gy) + 1e-6), 0, 1)
    band = np.clip(1 - d / width, 0, 1) * facing * m
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.0)
    streak = 0.5 + 0.5 * np.sin((X * 0.8 + Y * 0.6) * 0.35 + 3 * grain)
    k = (band * amount * (0.55 + 0.45 * streak))[..., None]
    col = np.array(color, np.float32)[None, None] / 255
    img[..., :3] = img[..., :3] * (1 - k) + col * k


def solid_black(f, regions, dark=0.45):
    """Inside each region (x, y, rx, ry): find the drawn black shape (dark, grey pixels connected to the
    middle), fill its holes (glare spots) and paint it flat near-black; colored lines on it (red
    cracks) stay. Only the shape itself changes, not the skin around it."""
    h, w = f.shape[:2]
    hsv = cv2.cvtColor((np.clip(f[..., :3], 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2] / 255
    for (ex, ey, rx, ry) in regions:
        x0, x1 = int(max(0, ex - 1.7 * rx)), int(min(w, ex + 1.7 * rx))
        y0, y1 = int(max(0, ey - 1.7 * ry)), int(min(h, ey + 1.7 * ry))
        dk = ((v[y0:y1, x0:x1] < dark) & (hsv[y0:y1, x0:x1, 1] < 90)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dk)
        cx, cy = int(ex - x0), int(ey - y0)
        k = lab[cy, cx] if lab[cy, cx] > 0 else (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else 0)
        if k == 0:
            continue
        m = (lab == k).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        fill_ = np.pad(m, 1)  # a free border, so everything outside the shape is one region
        ff = np.zeros((fill_.shape[0] + 2, fill_.shape[1] + 2), np.uint8)
        cv2.floodFill(fill_, ff, (0, 0), 1)
        m = m | (1 - fill_[1:-1, 1:-1])  # holes filled
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
        keep = np.clip((hsv[y0:y1, x0:x1, 1] - 90) / 50, 0, 1)  # saturated lines on top stay
        k_ = (m * (1 - keep))[..., None]
        reg = f[y0:y1, x0:x1, :3]
        rng = np.random.default_rng(int(ex))
        ink = 0.075 + 0.02 * cv2.GaussianBlur(rng.normal(0, 1, m.shape).astype(np.float32), (0, 0), 3)[..., None]
        reg[:] = reg * (1 - k_) + ink * k_
    return f


def body(c):
    im, alpha = load_page(c)
    h, w = im.shape[:2]
    pad, ext, bottom = c["pad"], c["ext"], c["bottom"]
    f = np.dstack([im.astype(np.float32) / 255, alpha])
    f = f[:bottom]
    if c.get("dark_eyes"):  # solid black marker shapes (eye sockets): the flash glare filled back in, exactly inside the drawn shape
        f = solid_black(f, [(ex, ey, rx, ry) for (ex, ey, rx, ry) in c["head"]["eyes"]])
    if c.get("shirt_top"):  # the phone flash left a grey glare on the black shirt: subtract it (smooth), letters kept
        s0_, s1_ = c["shirt"]
        y0 = c["shirt_top"]
        reg = f[y0:bottom, s0_:s1_, :3]
        hsv = cv2.cvtColor((reg * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        plain = ((hsv[..., 1] < 60) & (reg.mean(2) < 0.9)).astype(np.float32)  # shirt cloth (not letters)
        shirt_c = np.array(c.get("shirt_bgr", (30, 26, 28)), np.float32) / 255
        excess = np.clip(reg - shirt_c, 0, None) * plain[..., None]
        num = cv2.GaussianBlur(excess, (0, 0), 18)
        den = cv2.GaussianBlur(plain, (0, 0), 18)[..., None]
        glare = num / np.maximum(den, 1e-3)
        ramp = np.clip((np.arange(bottom - y0, dtype=np.float32) - 20) / 60, 0, 1)[:, None, None]  # not the neck
        fade = np.clip(np.minimum(np.arange(s1_ - s0_), s1_ - s0_ - 1 - np.arange(s1_ - s0_)) / 25, 0, 1)[None, :, None]
        reg -= glare * ramp * fade
        np.clip(reg, 0, 1, out=reg)
    H, Wb = bottom + ext, w + 2 * pad
    X = lambda x: x + pad  # noqa: E731
    big = np.zeros((H, Wb, 4), np.float32)
    big[:bottom, pad:pad + w] = f
    # below the page: plain jacket (white + grain) where the jacket reaches the page edge, the
    # black shirt between the lapels; only the real lines (lapel edges, outer edges) continue, as
    # strokes, below
    s0, s1 = c["shirt"]
    reach = (np.median(f[bottom - 30:bottom - 8, :, 3], axis=0) > 0.5).astype(np.float32)
    jt = jacket_tex(ext + 40, w, c["jacket_bgr"], 3)
    shirt_col = np.zeros(w, np.float32)
    shirt_col[s0:s1] = 1
    shirt = np.array(c.get("shirt_bgr", (30, 26, 28)), np.float32) / 255
    blk = np.zeros((ext + 40, w, 4), np.float32)
    blk[..., :3] = jt * (1 - shirt_col[None, :, None]) + shirt * shirt_col[None, :, None]
    blk[..., 3] = reach[None, :]
    k = np.clip(np.arange(40, dtype=np.float32) / 40, 0, 1)[:, None, None]
    big[bottom - 40:bottom, pad:pad + w] = big[bottom - 40:bottom, pad:pad + w] * (1 - k) + blk[:40] * k
    big[bottom:, pad:pad + w] = blk[40:]
    m_jk = np.zeros(big.shape[:2], np.float32)
    m_jk[bottom - 20:, pad:pad + w] = reach[None, :] * (1 - shirt_col[None, :])
    crayon(big, m_jk, (1, 0), c["shade_bgr"], 40, 0.5, 11)
    # hem: the jacket stops ext - 40 rows below the page, the shirt a little lower (tucked look)
    hem = bottom + ext - 60
    big[hem:, :, 3] = 0
    # sides: the jacket continues past the left and right page edges
    ext_l = np.zeros_like(big)
    a_, b_ = c["right_edge"]
    yl, yr = c["side_from"]
    xr = lambda y: a_ + b_ * y - 6  # noqa: E731
    left = [(X(2), yl), (X(-40), yl + 50), (X(-80), yl + 140), (X(-95), yl + 300), (X(-90), hem), (X(2), hem)]
    right = [(X(xr(yr)), yr), (X(xr(yr)) + 40, yr + 50), (X(xr(yr)) + 80, yr + 140), (X(xr(yr)) + 95, yr + 300), (X(xr(yr)) + 90, hem),
             (X(xr(hem)), hem)]
    base = jacket_tex(H, Wb, c["jacket_bgr"], 5)
    ml = fill(ext_l, np.array(left, float), base, 20)
    mr = fill(ext_l, np.array(right, float), base, 21)
    crayon(ext_l, ml, (-1, 0), c["shade_bgr"], 45, 0.55, 7)
    crayon(ext_l, mr, (1, 0), c["shade_bgr"], 45, 0.55, 8)
    out = ext_l.copy()
    a = big[..., 3:4]
    out[..., :3] = out[..., :3] * (1 - a) + big[..., :3] * a
    out[..., 3:4] = out[..., 3:4] * (1 - a) + a
    # the page's right edge cut through the jacket at a slant: cover the gap under the slant
    for y in range(yr, bottom):
        x = int(X(xr(y)))
        gap = out[y, x - 8:x + 4]
        gap[..., :3] = gap[..., :3] * gap[..., 3:4] + base[y, x - 8:x + 4, :3] * (1 - gap[..., 3:4])
        gap[..., 3] = 1.0
    # ink: outer side lines, the hem, the lapel edges continued
    stroke(out, np.array(left[:5], float), 11, seed=31)
    stroke(out, np.array(right[:5], float), 11, seed=32)
    stroke(out, [(X(-90), hem), (X(s0 - 4), hem - 6)], 11, seed=33)
    stroke(out, [(X(s1 + 4), hem - 6), (X(xr(hem)) + 90, hem)], 11, seed=34)
    stroke(out, [(X(s0 - 2), bottom - 30), (X(s0 - 4), hem - 6)], 10, seed=35)
    stroke(out, [(X(s1 + 2), bottom - 30), (X(s1 + 4), hem - 6)], 10, seed=36)
    # shirt bottom edge (black shirt, under the jacket)
    stroke(out, [(X(s0), hem + 30), (X(s1), hem + 30)], 9, seed=37)
    out[hem:hem + 34, X(s0):X(s1)] = np.concatenate([np.repeat(np.array([[[0.12, 0.10, 0.11]]], np.float32), 1, 0).repeat(34, 0).repeat(X(s1) - X(s0), 1),
                                                     np.ones((34, X(s1) - X(s0), 1), np.float32)], 2)
    stroke(out, [(X(s0), hem + 32), (X(s1), hem + 32)], 9, seed=38)
    return np.clip(out, 0, 1)


def body_hoodie(c):
    """A figure with an inked head and a pencil-sketched hoodie: the head cut along its ink, the hoodie
    from a traced outline; the pencil lines darkened a little (so they read at video size); the
    hoodie continued below the page."""
    im = photo.flat_light(cv2.imread(f"{ROOT}/{c['src']}"))
    x0, y0, x1, y1 = c["crop"]
    h, w = y1 - y0, x1 - x0
    page = im[y0:y1, x0:x1].copy()
    bx0, by0, bx1, by1 = c["head_box"]
    hb = im[by0:by1, bx0:bx1]
    hh, hw = hb.shape[:2]
    seeds = [(x, 2) for x in range(0, hw, 10)] + [(2, y) for y in range(0, hh, 10)] + [(hw - 3, y) for y in range(0, hh, 10)] + \
            [(x, hh - 3) for x in range(0, hw, 10)]
    head_a = photo.cut_figure(hb, seeds, tol=26)
    alpha = np.zeros((h, w), np.float32)
    alpha[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = head_a
    hood = np.zeros((h * 4, w * 4), np.uint8)
    cv2.fillPoly(hood, [np.round((np.array(c["hoodie"], float) - [x0, y0]) * 4).astype(np.int32)], 255, cv2.LINE_AA)
    hood = cv2.resize(hood, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    alpha = np.maximum(alpha, hood)
    alpha[:max(0, by0 + 8 - y0)] = 0  # nothing above the head box (letters of the band name)
    f = page.astype(np.float32) / 255
    # pencil lines on the hoodie: darker, a little bluish-grey like the page
    g = f.mean(2)
    head_full = np.zeros((h, w), np.float32)
    head_full[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = head_a
    pencil = np.clip((0.93 - g) / 0.25, 0, 1)
    line = np.array(c["line_bgr"], np.float32) / 255
    base = np.array(c["jacket_bgr"], np.float32) / 255
    k = (hood * (1 - head_full))[..., None]  # the head keeps its drawing
    f = f * (1 - k) + (base * (1 - pencil[..., None]) + line * pencil[..., None]) * k
    # the outline of the hoodie (traced) as a pencil line
    out = np.zeros((h + c["ext"], w + 2 * c["pad"], 4), np.float32)
    pad = c["pad"]
    out[:h, pad:pad + w, :3] = f
    out[:h, pad:pad + w, 3] = alpha
    # below the page: hoodie white, the side lines continue
    reach = (alpha[h - 20:h - 4].mean(0) > 0.5).astype(np.float32)
    jt = jacket_tex(c["ext"], w, c["jacket_bgr"], 4)
    out[h:, pad:pad + w, :3] = jt
    out[h:, pad:pad + w, 3] = reach[None, :]
    hem = h + c["ext"] - 40
    out[hem:, :, 3] = 0
    P = np.array(c["hoodie"], float) - [x0 - pad, y0]
    side = c.get("hoodie_sides", (6, 8))  # outline: the left side up to the hood, the right side from the hood
    stroke(out, P[:side[0]], 6, seed=60, color=c["line_bgr"])
    stroke(out, P[side[1]:], 6, seed=59, color=c["line_bgr"])
    xs = np.nonzero(reach)[0]
    stroke(out, [(xs[0] + pad + 2, h - 5), (xs[0] + pad - 4, hem)], 6, seed=61, color=c["line_bgr"])
    stroke(out, [(xs[-1] + pad - 2, h - 5), (xs[-1] + pad + 4, hem)], 6, seed=62, color=c["line_bgr"])
    stroke(out, [(xs[0] + pad - 4, hem), (xs[-1] + pad + 4, hem)], 6, seed=63, color=c["line_bgr"])
    rib = hem - 70  # the hoodie's waistband
    stroke(out, [(xs[0] + pad, rib), (xs[-1] + pad, rib)], 5, seed=64, color=c["line_bgr"])
    return np.clip(out, 0, 1)


def sleeve(c, length, r0, r1, joint0, cuff=False, seed=0):
    """One arm part: a sleeve with crayon shading along one side and an ink outline. The upper arm
    is a tapered capsule (its top hides behind the body). The forearm widens a little toward an open,
    flat cuff (the hand comes out of it, drawn under the sleeve), with a cuff band and a crease."""
    wdt = int(2 * max(r0, r1) + 60)
    h = int(length + 2 * max(r0, r1) + 60)
    img = np.zeros((h, wdt, 4), np.float32)
    A = (wdt / 2, joint0)
    B = (wdt / 2, joint0 + length)
    lw, lc = (6, c["line_bgr"]) if c.get("line_bgr") else (10, None)
    if not cuff:
        out = capsule(A, r0, B, r1, bulge=4)
        m = fill(img, out, jacket_tex(h, wdt, c["jacket_bgr"], seed), seed)
        crayon(img, m, (1, 0.2), c["shade_bgr"], 0.45 * r0, 0.6, seed + 1)
        stroke(img, out, lw, closed=True, seed=seed + 2, color=lc)
        # the top grows out of the jacket's shoulder: no cap, the sleeve fades in over its first part
        yy = np.arange(h, dtype=np.float32)[:, None]
        img[..., 3] *= np.clip((yy - (A[1] - 0.1 * r0)) / (0.9 * r0), 0, 1)
        return img, A, B
    # forearm: elbow end round, the wrist end an open flat cuff
    cx, y0, y1 = wdt / 2, joint0, joint0 + length
    left = [(cx - r0 - 2 * math.sin(math.pi * t) - (r1 - r0) * t, y0 + length * t) for t in np.linspace(0, 1, 40)]
    right = [(cx + r0 + 2 * math.sin(math.pi * t) + (r1 - r0) * t, y0 + length * t) for t in np.linspace(1, 0, 40)]
    cap = [(cx + r0 * math.cos(a_), y0 - r0 * math.sin(a_)) for a_ in np.linspace(0, math.pi, 30)]
    poly = np.array(left + [(cx + r1, y1)] + right + cap[::-1], float)
    poly = np.array(left + [(cx - r1, y1), (cx + r1, y1)] + right + cap, float)
    m = fill(img, poly, jacket_tex(h, wdt, c["jacket_bgr"], seed), seed)
    crayon(img, m, (1, 0.2), c["shade_bgr"], 0.45 * r0, 0.6, seed + 1)
    stroke(img, np.array(right + cap + left, float), lw, seed=seed + 2, color=lc)  # the sides and the elbow end
    stroke(img, [(cx - r1, y1), (cx + r1, y1)], lw, seed=seed + 6, color=lc)  # the cuff's open edge
    band = y1 - 0.16 * length  # the cuff band
    stroke(img, [(cx - r1 + 4, band), (cx, band + 6), (cx + r1 - 4, band)], max(3, lw - 3), seed=seed + 3, color=lc)
    return img, A, (cx, y1)


def load_rig(name):
    return json.load(open(f"{ROOT}/public/rigs/{name}.json"))


def main(name):
    c = load_rig(name)
    out = f"{ROOT}/public/parts/{name}"
    os.makedirs(out, exist_ok=True)
    b = body_hoodie(c) if c.get("kind") == "hoodie" else body(c)
    save = lambda p, im: cv2.imwrite(p, np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8))  # noqa: E731
    save(f"{out}/body.png", b)
    ar = c["arm"]
    r = ar["width"] / 2
    up, S, E = sleeve(c, ar["upper"], r, 0.9 * r, r + 20, seed=40)
    fo, E2, Wr = sleeve(c, ar["fore"], 0.82 * r, 0.95 * r, 0.82 * r + 20, cuff=True, seed=50)
    save(f"{out}/upper_arm.png", up)
    save(f"{out}/forearm.png", fo)
    pad = c["pad"]
    hd = dict(c["head"])
    if c.get("crop"):  # page coords -> crop coords
        cx0, cy0 = c["crop"][:2]
        hd = dict(hd, head_ells=[(x - cx0, y - cy0, rx, ry) for x, y, rx, ry in hd["head_ells"]],
                  neck=(hd["neck"][0] - cx0, hd["neck"][1] - cx0, hd["neck"][2] - cy0, hd["neck"][3] - cy0),
                  face=(hd["face"][0] - cx0, hd["face"][1] - cy0) + tuple(hd["face"][2:]),
                  pivot=(hd["pivot"][0] - cx0, hd["pivot"][1] - cy0), mouth=(hd["mouth"][0] - cx0, hd["mouth"][1] - cy0) + tuple(hd["mouth"][2:]),
                  chin=hd["chin"] - cy0, eyes=[(x - cx0, y - cy0, rx, ry) for x, y, rx, ry in hd["eyes"]], hair_from=hd["hair_from"] - cy0)
        if "lid_color" in hd:
            hd["lid_color"] = (hd["lid_color"][0] - cx0, hd["lid_color"][1] - cy0)
        c = dict(c, shoulders={k: (v[0] - cx0, v[1] - cy0) for k, v in c["shoulders"].items()})
    j = dict(credit=c["credit"], pad=pad, src_h=b.shape[0], head=hd,
             hand=dict(skin_bgr=list(c["skin_bgr"]), length=c["hand_length"]),
             upper_arm=dict(file="upper_arm.png", shoulder=[S[0], S[1]], elbow=[E[0], E[1]]),
             forearm=dict(file="forearm.png", elbow=[E2[0], E2[1]], wrist=[Wr[0], Wr[1] - 0.04 * ar["fore"]]),
             shoulders=dict(left=[c["shoulders"]["left"][0] + pad, c["shoulders"]["left"][1]],
                            right=[c["shoulders"]["right"][0] + pad, c["shoulders"]["right"][1]]),
             arm_width=ar["width"], play=c.get("play", {}), blink=c.get("blink", True))
    json.dump(j, open(f"{out}/parts.json", "w"), indent=1)
    print(out, b.shape)


if __name__ == "__main__":
    main(sys.argv[1])
