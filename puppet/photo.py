"""Ben's drawings arrive as phone photos: a sheet of paper on a table, a little skewed, with uneven
light. page() finds the sheet and warps it flat and upright; flat_light() evens out the light;
cut_figure() removes the paper around a figure (flood fill from the parts of the page border
where the figure does not touch it).

  python3 puppet/photo.py page <photo> <out.png>        (check the result by eye)
"""
import sys

import cv2
import numpy as np


def page(img, out_h=2000):
    """The largest bright quadrilateral (the sheet), warped to a rectangle. Returns (page, quad)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 3)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    quad = None
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(ap) == 4:
            quad = ap[:, 0, :].astype(np.float32)
            break
    if quad is None:  # fall back to the min-area rectangle
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    # order: top-left, top-right, bottom-right, bottom-left
    s, d = quad.sum(1), np.diff(quad, axis=1)[:, 0]
    q = np.array([quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]], np.float32)
    wt = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    ht = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    out_w = int(round(out_h * wt / ht))
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32))
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), q


def flat_light(img, cell=24):
    """Even out the light: estimate the paper's brightness from paper pixels only (bright, grey),
    interpolate it under the drawing, and divide, so the paper becomes even white and dark areas
    stay dark (a plain morphological closing would lighten big black shapes)."""
    f = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = f.mean(2)
    loc = cv2.dilate(g, np.ones((41, 41), np.uint8))  # the local paper level is near the local max
    paperish = (hsv[..., 1] < 45) & (g > 0.82 * loc)
    h, w = g.shape
    gh, gw = (h + cell - 1) // cell, (w + cell - 1) // cell
    acc = np.zeros((gh, gw, 3), np.float32)
    cnt = np.zeros((gh, gw), np.float32)
    ys, xs = np.nonzero(paperish)
    np.add.at(acc, (ys // cell, xs // cell), f[ys, xs])
    np.add.at(cnt, (ys // cell, xs // cell), 1)
    ok = cnt > 0.15 * cell * cell
    small = np.where(ok[..., None], acc / np.maximum(cnt, 1)[..., None], 0)
    # fill cells under the drawing from their paper neighbours (normalized blur, growing)
    m = ok.astype(np.float32)
    est, wsum = small * m[..., None], m.copy()
    for r in (1, 2, 4, 8, 16, 32):
        b = cv2.GaussianBlur(small * m[..., None], (0, 0), r)
        bw = cv2.GaussianBlur(m, (0, 0), r)
        fillv = b / np.maximum(bw, 1e-6)[..., None]
        need = wsum < 0.5
        est[need] = fillv[need]
        wsum = np.maximum(wsum, (bw > 0.02).astype(np.float32))
    est = cv2.GaussianBlur(est, (0, 0), 1.5)
    bg = cv2.resize(est, (w, h), interpolation=cv2.INTER_CUBIC)
    out = f / np.maximum(bg, 1) * 245
    return np.clip(out, 0, 255).astype(np.uint8)


def cut_figure(img, seeds, tol=30, ink_close=5, keep_largest=True):
    """Alpha for the figure: flood the paper from seed points (on the page border where the figure
    does not touch it). Ink lines are thickened a little first so small gaps do not leak."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (g < 110).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((ink_close, ink_close), np.uint8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper = np.median(lab[g > np.percentile(g, 70)], axis=0)
    near = (np.sqrt(((lab - paper) ** 2).sum(2)) < tol).astype(np.uint8) & (1 - ink)
    reach = np.zeros_like(near)
    h, w = near.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    work = near * 255
    for (x, y) in seeds:
        if work[y, x] == 255:
            cv2.floodFill(work, mask, (int(x), int(y)), 128)
    reach = work == 128
    fg = (~reach).astype(np.uint8)
    if keep_largest:
        n, lab_, st, _ = cv2.connectedComponentsWithStats(fg)
        fg = (lab_ == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # give back the ink dilation: the flood stopped ink_close/2 px outside the outline
    fg = cv2.erode(fg, np.ones((max(1, ink_close // 2), max(1, ink_close // 2)), np.uint8))
    return cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 0.8)


if __name__ == "__main__":
    if sys.argv[1] == "page":
        im = cv2.imread(sys.argv[2])
        pg, q = page(im)
        cv2.imwrite(sys.argv[3], flat_light(pg))
        print("quad", q.round().tolist(), "->", pg.shape)
