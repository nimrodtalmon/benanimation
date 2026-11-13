"""Cut Ben's mouth poses out of his photographed sheet (drawings/ben/mouths-sheet.jpg):
flatten the paper's light falloff, cut each mouth by its closed ink outline (plus the little
corner strokes), turn it level, and save BGRA sprites.

  python3 puppet/ben_mouths.py -> public/parts/distor/mouth_*.png + mouths.json
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/public/parts/distor"
# crop boxes on the sheet (x0, y0, x1, y1)
MOUTHS = dict(
    wide=(470, 470, 1060, 750),   # flat, teeth clenched / slightly open
    open=(670, 980, 1240, 1330),  # wide open, big teeth
)
# the inside of each mouth outline, traced by hand on the sheet (the photo's outline has gaps)
INSIDE = dict(
    wide=[(497, 537), (560, 530), (650, 527), (750, 533), (850, 553), (930, 578), (990, 607), (1018, 640), (1025, 668), (1012, 690),
          (950, 698), (850, 688), (750, 668), (670, 640), (600, 605), (540, 568)],
    open=[(700, 1095), (720, 1060), (770, 1030), (860, 1010), (960, 1015), (1060, 1040), (1140, 1080), (1190, 1130), (1205, 1180),
          (1185, 1225), (1145, 1255), (1060, 1276), (950, 1284), (850, 1274), (775, 1248), (725, 1205), (703, 1160)],
)


def flatten(img):
    """Divide by the paper's brightness so the paper becomes even white."""
    g = img.astype(np.float32)
    bg = cv2.dilate(g, np.ones((41, 41), np.uint8))  # paper is the brightest thing around
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    out = np.clip(g / np.maximum(bg, 1) * 245, 0, 255)
    return out.astype(np.uint8)


def cut(name, box):  # noqa: C901
    sheet = cv2.imread(f"{ROOT}/drawings/ben/mouths-sheet.jpg")
    x0, y0, x1, y1 = box
    c = flatten(sheet[y0:y1, x0:x1])
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    poly = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(poly, [np.array(INSIDE[name], np.int32) - [x0, y0]], 1)
    # shrink the hand-traced shape well inside, then grow it out to the ink outline (never across ink)
    light = (gray > 140).astype(np.uint8)
    body = cv2.erode(poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(26):
        body = np.maximum(body, cv2.dilate(body, k3) & light)
    body &= cv2.dilate(poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # no leaks through gaps
    body = np.maximum(body, cv2.dilate(body, k3) & (1 - light))  # plus the ink outline itself
    # clean the colors: deep blacks neutral, light teeth a touch warmer and even
    cf = c.astype(np.float32)
    lum = cf.mean(2, keepdims=True)
    red = (cf[..., 2:3] - cf[..., 1:2]) > 40
    blk = np.clip((150 - lum) / 45, 0, 1) * ~red
    cf = cf * (1 - blk) + (lum * 0.25 + np.array([18, 16, 20], np.float32)) * blk
    c = np.clip(cf, 0, 255).astype(np.uint8)
    # alpha: inside the outline everything; outside only the ink strokes (corner flicks, lip line)
    dark = np.clip((235 - gray.astype(np.float32)) / 120, 0, 1)
    near = cv2.dilate(body, np.ones((61, 61), np.uint8)).astype(np.float32)
    a = np.maximum(cv2.GaussianBlur(body.astype(np.float32), (0, 0), 0.8), dark * near)
    a[dark < 0.15] *= body[dark < 0.15]
    # level it: rotate so the mouth's long axis is horizontal (search the angle; expand the canvas)
    rgba = np.dstack([c, (a * 255).astype(np.uint8)])
    pad = max(c.shape[:2]) // 2
    rgba = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    bod = cv2.copyMakeBorder(body * 255, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = bod.shape
    ys, xs = np.nonzero(bod)
    cx, cy = float(xs.mean()), float(ys.mean())
    best = None
    for theta in np.arange(-40, 41, 1.0):
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        b2 = cv2.warpAffine(bod, M, (W, H), flags=cv2.INTER_NEAREST)
        yy = np.nonzero(b2.any(1))[0]
        tall = yy.max() - yy.min()
        if best is None or tall < best[0]:
            best = (tall, theta)
    M = cv2.getRotationMatrix2D((cx, cy), best[1], 1.0)
    r = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    bm = cv2.warpAffine(bod, M, (W, H), flags=cv2.INTER_NEAREST) > 128
    ys, xs = np.nonzero(r[..., 3] > 10)
    r = r[ys.min() - 2:ys.max() + 3, xs.min() - 2:xs.max() + 3]
    bm = bm[ys.min() - 2:ys.max() + 3, xs.min() - 2:xs.max() + 3]
    # the inner mouth box (outline only) gives the anchors used for placement
    by, bx = np.nonzero(bm)
    info = dict(file=f"mouth_{name}.png", w=int(r.shape[1]), h=int(r.shape[0]),
                left=int(bx.min()), right=int(bx.max()), top=int(by.min()), bottom=int(by.max()))
    cv2.imwrite(f"{OUT}/mouth_{name}.png", r)
    return info


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    info = {k: cut(k, b) for k, b in MOUTHS.items()}
    json.dump(info, open(f"{OUT}/mouths.json", "w"), indent=1)
    print(info)
