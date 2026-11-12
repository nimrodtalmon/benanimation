"""Split the Distor front cutout into a head layer and a body layer, plus an eye mask.

The body layer keeps a band of neck skin under the jaw, so a small head tilt
shows neck instead of a hole. The eye mask marks the eye whites (for blinking).

  python scripts/split_front.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

SRC = "public/chars/distor-front.png"
# Head outline in cutout coordinates: everything above the jaw (incl. ears and hair).
HEAD = [(0, 0), (1150, 0), (1150, 790), (810, 790), (685, 940), (540, 1025), (480, 1025), (365, 950), (230, 800), (0, 790)]
# Neck skin band behind the jaw (same jaw line, pushed up).
NECK_BAND = [(300, 830), (365, 905), (480, 985), (540, 985), (685, 895), (750, 830), (765, 860), (685, 945), (540, 1030), (480, 1030), (365, 955), (290, 870)]
NECK_COLOR_AT = (470, 1080)
EYES = [(305, 500, 535, 615), (620, 505, 855, 650)]


def main():
    im = Image.open(SRC).convert("RGBA")
    w, h = im.size
    head_mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(head_mask).polygon(HEAD, fill=255)
    head_mask = head_mask.filter(ImageFilter.GaussianBlur(1))

    a = np.asarray(im).copy()
    hm = np.asarray(head_mask).astype(np.float32) / 255

    head = a.copy()
    head[..., 3] = (head[..., 3] * hm).astype(np.uint8)
    Image.fromarray(head).save("public/chars/distor-front-head.png")

    body = a.copy()
    body[..., 3] = (body[..., 3] * (1 - hm)).astype(np.uint8)
    body_im = Image.fromarray(body)
    neck = tuple(a[NECK_COLOR_AT[1], NECK_COLOR_AT[0], :3]) + (255,)
    band = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(band).polygon(NECK_BAND, fill=neck)
    out = Image.alpha_composite(band, body_im)
    out.save("public/chars/distor-front-body.png")

    # Eye whites: light pixels inside the eye boxes, largest blob per eye.
    rgb = a[..., :3].astype(np.int16)
    light = (rgb.min(axis=2) > 150)
    mask = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in EYES:
        box = np.zeros_like(light)
        box[y0:y1, x0:x1] = light[y0:y1, x0:x1]
        lab, n = ndi.label(box)
        if n:
            sizes = ndi.sum(box, lab, range(1, n + 1))
            mask |= lab == 1 + int(np.argmax(sizes))
    mask = ndi.binary_dilation(mask, iterations=4)  # cover the anti-aliased rim
    eyes = np.zeros((h, w, 4), np.uint8)
    eyes[..., :3] = 255
    eyes[..., 3] = mask * 255  # alpha mask (works with CSS mask-image)
    Image.fromarray(eyes).save("public/chars/distor-front-eyes.png")
    print("neck color", neck, "eye mask px", int(mask.sum()))


if __name__ == "__main__":
    main()
