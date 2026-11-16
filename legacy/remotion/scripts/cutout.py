"""Cut a hand-drawn character out of a scanned page.

The drawing must have closed black outlines. The paper outside the outlines is
found by flood fill from seed points on the paper, with small gaps in the
outlines closed first so the fill does not leak inside the character.

Usage:
  python scripts/cutout.py --all            # every drawing in drawings/cutouts.json
  python scripts/cutout.py SCAN.png OUT.png --crop x0,y0,x1,y1 [--seed x,y ...]

  --crop   rectangle of the paper inside the scan (pixels in SCAN)
  --seed   points on the paper (in cropped coordinates); default: two
           points near the right edge
  --gap    outline gaps up to this many pixels are closed (default 6)
  --edge-walls  page edges that cut through the character (e.g. left,bottom),
                so the fill cannot sneak in along them
  --all-pieces  keep every piece (default keeps only the largest: drops
                signatures and specks)
"""
import argparse

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scan", nargs="?")
    p.add_argument("out", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--crop")
    p.add_argument("--seed", action="append", default=[])
    p.add_argument("--gap", type=int, default=6)
    p.add_argument("--dark", type=int, default=110, help="ink threshold (0-255)")
    p.add_argument("--all-pieces", action="store_true")
    p.add_argument("--edge-walls", default="", help="page edges the character is cut by, e.g. left,bottom")
    a = p.parse_args()
    if a.all:
        import json
        for name, c in json.load(open("drawings/cutouts.json")).items():
            cut(c["scan"], f"public/chars/{name}.png", c["crop"], c.get("seed", []),
                c.get("gap", 6), c.get("dark", 110), c.get("all_pieces", False), c.get("edge_walls", ""),
                c.get("paper_min", 200))
    else:
        cut(a.scan, a.out, a.crop, a.seed, a.gap, a.dark, a.all_pieces, a.edge_walls)


def cut(scan, out_path, crop, seed, gap, dark, all_pieces, edge_walls, paper_min=200):
    x0, y0, x1, y1 = map(int, crop.split(","))
    img = Image.open(scan).convert("RGB").crop((x0, y0, x1, y1))
    rgb = np.asarray(img).astype(np.int16)
    h, w = rgb.shape[:2]

    # Ink = dark pixels. Thicken it to close small gaps in the outlines.
    ink = rgb.max(axis=2) < dark
    for edge in filter(None, edge_walls.split(",")):
        sl = {"left": np.s_[:, :2], "right": np.s_[:, -2:], "top": np.s_[:2, :], "bottom": np.s_[-2:, :]}[edge]
        ink[sl] = True
    walls = ndi.binary_dilation(ink, iterations=gap)

    # Paper = light, low-saturation pixels that are not walls.
    hi, lo = rgb.max(axis=2), rgb.min(axis=2)
    paper = (hi > paper_min) & ((hi - lo) < 40) & ~walls

    seeds = [tuple(map(int, s.split(","))) for s in seed] or [(w - 30, h // 2), (w - 30, 60)]
    labels, _ = ndi.label(paper)
    bg_ids = {labels[y, x] for x, y in seeds if labels[y, x] > 0}
    bg = np.isin(labels, list(bg_ids))
    # Give back the pixels eaten by the wall thickening.
    bg = ndi.binary_dilation(bg, iterations=gap - 1) & ~ink

    # Edge-wall pixels take the state of the first column/row past the wall.
    k = gap + 2
    for edge in filter(None, edge_walls.split(",")):
        if edge == "left": bg[:, :k] = bg[:, k:k + 1]
        if edge == "right": bg[:, -k:] = bg[:, -k - 1:-k]
        if edge == "top": bg[:k, :] = bg[k:k + 1, :]
        if edge == "bottom": bg[-k:, :] = bg[-k - 1:-k, :]
    fg = ~bg
    if not all_pieces:
        lab, n = ndi.label(fg)
        if n > 1:
            sizes = ndi.sum(fg, lab, range(1, n + 1))
            fg = lab == (1 + int(np.argmax(sizes)))
    fg = ndi.binary_fill_holes(fg)

    alpha = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    out = img.copy()
    out.putalpha(alpha)
    out.save(out_path)
    print(f"saved {out_path} {w}x{h}, foreground {fg.mean():.0%}")


if __name__ == "__main__":
    main()
