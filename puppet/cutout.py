"""Turning photos of Ben's drawings into sprites (reusable for every new drawing).

  ink_sprite(photo, box, out)     line art only (eyes closed, brows, folds): alpha = darkness
  paper_flatten(img)              even out the light falloff of a phone photo
  paper_cutout(img, paper_bgr)    remove a flat paper color around a colored drawing

python3 puppet/cutout.py         -> public/parts/distor/eye_closed.png
"""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paper_flatten(img, radius=41):
    """Divide by the local paper brightness (the brightest thing around) so the paper is even."""
    g = img.astype(np.float32)
    bg = cv2.dilate(g, np.ones((radius, radius), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), radius * 0.6)
    return np.clip(g / np.maximum(bg, 1) * 245, 0, 255).astype(np.uint8)


def ink_sprite(photo, box, out=None, ink=(20, 18, 22), lo=70, hi=200, blur=0.8):
    """Line art from a photo: flatten the paper, alpha from darkness (hi = paper, lo = full ink),
    color = Ben's marker black. Returns BGRA uint8, cropped to the ink."""
    img = cv2.imread(photo) if isinstance(photo, str) else photo
    x0, y0, x1, y1 = box
    c = paper_flatten(img[y0:y1, x0:x1])
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = np.clip((hi - gray) / (hi - lo), 0, 1)
    a = cv2.GaussianBlur(a, (0, 0), blur)
    a[a < 0.06] = 0
    ys, xs = np.nonzero(a > 0.1)
    a = a[ys.min() - 3:ys.max() + 4, xs.min() - 3:xs.max() + 4]
    rgba = np.zeros(a.shape + (4,), np.uint8)
    rgba[..., :3] = ink
    rgba[..., 3] = (a * 255).astype(np.uint8)
    if out:
        cv2.imwrite(out, rgba)
    return rgba


def paper_cutout(img, paper_bgr, tol=40):
    """Remove the paper (flood fill from the border over paper-colored pixels); keep the largest blob."""
    paper = np.array(paper_bgr, np.float32)
    d = np.sqrt(((img.astype(np.float32) - paper) ** 2).sum(2))
    bg = (d < tol).astype(np.uint8) * 255
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    h, w = bg.shape
    for x, y in [(x, 0) for x in range(0, w, 7)] + [(x, h - 1) for x in range(0, w, 7)] + [(0, y) for y in range(0, h, 7)] + [(w - 1, y) for y in range(0, h, 7)]:
        if bg[y, x] == 255:
            cv2.floodFill(bg, mask, (x, y), 128)
    fg = (bg != 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.float32)
    return np.dstack([img, (cv2.GaussianBlur(fg, (0, 0), 0.7) * 255).astype(np.uint8)])


if __name__ == "__main__":
    ink_sprite(f"{ROOT}/drawings/ben/eye-closed.jpg", (450, 235, 1365, 535), f"{ROOT}/public/parts/distor/eye_closed.png")
    print("ok")
