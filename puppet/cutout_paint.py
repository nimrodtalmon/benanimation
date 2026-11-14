"""Cut the two Revoy stand-in figures out of their backgrounds.
  python3 puppet/cutout_paint.py -> public/paint/pepper-sheet-front.png, public/paint/coriander-full.png
Pepper model sheet: white paper -> flood fill from the border over light pixels.
Coriander concept art: smooth gradient background -> GrabCut from a rectangle.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pepper():
    im = cv2.imread(f"{ROOT}/drawings/revoy/pepper-model-sheet.jpg")
    x0, y0, x1, y1 = 40, 60, 850, 1460  # front view, down to the knees
    c = im[y0:y1, x0:x1].copy()
    light = (c.min(2) > 200).astype(np.uint8)
    # everything light that touches the border (or the faint guide lines) is paper
    ff = light.copy() * 255
    mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    hh, ww = ff.shape
    border = [(x, 0) for x in range(ww)] + [(x, hh - 1) for x in range(ww)] + [(0, y) for y in range(hh)] + [(ww - 1, y) for y in range(hh)]
    for (x, y) in border:
        if ff[y, x] == 255:
            cv2.floodFill(ff, mask, (x, y), 128)
    bg = (ff == 128).astype(np.uint8)
    # the head drawings at the bottom right and the side sketch: keep only the largest blob
    fg = 1 - bg
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    fg = (lab == k).astype(np.uint8)
    fg[:, 790 - x0:] = 0  # side-view sketch strokes on the right
    # soft edge: alpha from how far the color is from paper white, inside a dilated mask
    soft = np.clip((255 - c.min(2).astype(np.float32)) / 40, 0, 1)
    core = cv2.erode(fg, np.ones((3, 3), np.uint8))
    a = np.maximum(core.astype(np.float32), soft * cv2.dilate(fg, np.ones((5, 5), np.uint8)))
    a = cv2.GaussianBlur(a, (0, 0), 0.8) * cv2.dilate(fg, np.ones((5, 5), np.uint8))
    out = np.dstack([c, (a * 255).astype(np.uint8)])
    cv2.imwrite(f"{ROOT}/public/paint/pepper-sheet-front.png", out)
    return (x0, y0)


def coriander():
    im = cv2.imread(f"{ROOT}/drawings/revoy/coriander-concept.jpg")
    h, w = im.shape[:2]
    s = 0.5
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = np.zeros(small.shape[:2], np.uint8)
    rect = (int(330 * s), int(130 * s), int((1070 - 330) * s), int((1900 - 130) * s))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    fg = (lab == k).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    # the background is a smooth gradient: estimate it from outside the figure, then alpha from
    # how far each pixel is from it (inside a band around the GrabCut edge)
    out_m = cv2.erode(1 - fg, np.ones((25, 25), np.uint8)).astype(np.float32)
    imf = im.astype(np.float32)
    num = cv2.GaussianBlur(imf * out_m[..., None], (0, 0), 60)
    den = cv2.GaussianBlur(out_m, (0, 0), 60)[..., None] + 1e-4
    bg = num / den
    d = np.sqrt(((imf - bg) ** 2).sum(2))
    soft = np.clip((d - 14) / 26, 0, 1)
    inner = cv2.erode(fg, np.ones((31, 31), np.uint8)).astype(np.float32)
    inner[1330:] = 0  # below the coat: the floor shows between the legs, trust the color distance
    band = cv2.dilate(fg, np.ones((9, 9), np.uint8)).astype(np.float32)
    a = np.maximum(inner, soft * band)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(a, (0, 0), 0.7)
    a[1780:] = 0  # floor shadow under the boots
    floor = (im.min(2) > 160).astype(np.float32)  # light floor between the legs
    floor = cv2.GaussianBlur(cv2.dilate(floor, np.ones((5, 5), np.uint8)), (0, 0), 2)
    a[1280:] *= 1 - floor[1280:]
    out = np.dstack([im, (a * 255).astype(np.uint8)])
    cv2.imwrite(f"{ROOT}/public/paint/coriander-full.png", out)


# ---------------------------------------------------------------- Pepper as a drummer: arms apart
PEP_ARM_L = [(250, 565), (205, 590), (190, 650), (180, 760), (175, 850), (140, 930), (100, 1000), (75, 1060), (45, 1140), (45, 1200),
             (80, 1250), (130, 1255), (175, 1200), (185, 1120), (200, 1060), (225, 980), (250, 900), (268, 820), (272, 720), (275, 640), (268, 580)]
PEP_TORSO_L = [(270, 520), (270, 640), (264, 720), (262, 890), (250, 990), (240, 1300)]
PEP_MID = 403  # the figure's middle (x), arms are mirrored around it


def pepper_parts():
    """Body without arms (the strip of vest behind each arm is inpainted) + one image per arm."""
    im = cv2.imread(f"{ROOT}/public/paint/pepper-sheet-front.png", cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]
    mir = lambda pts: [(2 * PEP_MID - x, y) for x, y in pts]  # noqa: E731
    out = f"{ROOT}/public/parts/pepper"
    os.makedirs(out, exist_ok=True)
    torso = np.zeros((h, w), np.uint8)
    cv2.fillPoly(torso, [np.array(PEP_TORSO_L + mir(PEP_TORSO_L)[::-1], np.int32)], 1)
    arms = np.zeros((h, w), np.uint8)
    body = im.copy()
    for side, poly in (("l", PEP_ARM_L), ("r", mir(PEP_ARM_L))):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
        arms |= m
        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)
        # the painted hand is replaced by a rigged hand: cut the arm at the end of the cuff
        el = np.array([215.0, 855.0]) if side == "l" else np.array([2 * PEP_MID - 215.0, 855.0])
        wr = np.array([135.0, 1085.0]) if side == "l" else np.array([2 * PEP_MID - 135.0, 1085.0])
        u = (wr - el) / np.linalg.norm(wr - el)
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        along = (X - wr[0]) * u[0] + (Y - wr[1]) * u[1]
        a *= np.clip((-2 - along) / 3, 0, 1)
        arm = im.copy()
        arm[..., 3] = (arm[..., 3] * a).astype(np.uint8)
        cv2.imwrite(f"{out}/arm_{side}.png", arm)
    # vest under the arms: inpaint (from the vest side), keep it only inside the torso outline
    fill = cv2.inpaint(im[..., :3], cv2.dilate(arms, np.ones((5, 5), np.uint8)), 12, cv2.INPAINT_TELEA)
    under = (arms & torso).astype(np.float32)
    under = cv2.GaussianBlur(under, (0, 0), 1.5)[..., None]
    body[..., :3] = (im[..., :3] * (1 - under) + fill * under).astype(np.uint8)
    keep = np.clip(1 - cv2.GaussianBlur((arms & (1 - torso)).astype(np.float32), (0, 0), 1.2), 0, 1)
    body[..., 3] = (im[..., 3] * keep).astype(np.uint8)
    body[..., 3] = np.maximum(body[..., 3], (under[..., 0] * 255).astype(np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats((body[..., 3] > 20).astype(np.uint8))
    big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    body[..., 3] = body[..., 3] * (lab == big)
    cv2.imwrite(f"{out}/body.png", body)
    J = dict(
        credit='Painting: Pepper model sheet by David Revoy (peppercarrot.com), CC BY 4.0, cut out and animated',
        arm_l=dict(file="arm_l.png", shoulder=[235, 650], elbow=[215, 855], wrist=[135, 1085], grip=[112, 1150]),
        arm_r=dict(file="arm_r.png", shoulder=[2 * PEP_MID - 235, 650], elbow=[2 * PEP_MID - 215, 855], wrist=[2 * PEP_MID - 135, 1085], grip=[2 * PEP_MID - 112, 1150]),
        arm_width=75, hand=dict(skin_bgr=[124, 159, 220], length=150))
    json.dump(J, open(f"{out}/parts.json", "w"), indent=1)


# ---------------------------------------------------------------- Pepper walking: legs below the skirt
PEP_HEM = 1392  # sheet y just below the skirt hem
PEP_LEG_J = dict(hip=(345, 1110), knee=(330, 1500), ankle=(300, 1975), sole=2042)  # sheet coords (viewer-left leg)


def pepper_legs():
    """Her viewer-left leg from the model sheet (the other one is covered by a head study, so the
    right leg is its mirror), and a body without the legs (cut at the hem)."""
    im = cv2.imread(f"{ROOT}/drawings/revoy/pepper-model-sheet.jpg")
    X0, Y0, X1, Y1 = 150, 1050, 470, 2070
    c = im[Y0:Y1, X0:X1].copy()
    hh, ww = c.shape[:2]
    light = (c.min(2) > 205).astype(np.uint8) * 255
    mask = np.zeros((hh + 2, ww + 2), np.uint8)
    for (x, y) in [(x, hh - 1) for x in range(ww)] + [(0, y) for y in range(hh)] + [(ww - 1, y) for y in range(hh)]:
        if light[y, x] == 255:
            cv2.floodFill(light, mask, (x, y), 128)
    fg = (light != 128).astype(np.uint8)
    fg[: PEP_HEM - Y0 + 12] = 0  # the hem's shadow line
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    fg = (lab == k).astype(np.uint8)
    soft = np.clip((255 - c.min(2).astype(np.float32)) / 40, 0, 1)
    a = np.maximum(cv2.erode(fg, np.ones((3, 3), np.uint8)).astype(np.float32), soft * cv2.dilate(fg, np.ones((5, 5), np.uint8)))
    a = cv2.GaussianBlur(a, (0, 0), 0.8) * cv2.dilate(fg, np.ones((5, 5), np.uint8))
    # fade in just under the hem (the skirt overlaps it)
    a[: PEP_HEM - Y0 + 24] *= np.linspace(0, 1, PEP_HEM - Y0 + 24)[:, None] ** 4
    leg = np.dstack([c, (a * 255).astype(np.uint8)])
    # the thigh goes on under the skirt (so a swinging leg never shows its cut top)
    r0 = PEP_HEM - Y0 + 30
    top = PEP_LEG_J["hip"][1] - Y0 + 40
    leg[top:r0] = leg[r0]
    out = f"{ROOT}/public/parts/pepper"
    cv2.imwrite(f"{out}/leg_l.png", leg)
    cv2.imwrite(f"{out}/leg_r.png", leg[:, ::-1])
    J = json.load(open(f"{out}/parts.json"))
    mid = PEP_MID + 40  # sheet x of her middle
    lj = {k: [v[0] - X0, v[1] - Y0] for k, v in PEP_LEG_J.items() if k != "sole"}
    rj = {k: [ww - 1 - v[0], v[1]] for k, v in lj.items()}
    J["legs"] = dict(
        l=dict(file="leg_l.png", hip_body=[PEP_LEG_J["hip"][0] - 40, PEP_LEG_J["hip"][1] - 60], **lj),
        r=dict(file="leg_r.png", hip_body=[2 * mid - PEP_LEG_J["hip"][0] - 40, PEP_LEG_J["hip"][1] - 60], **rj))
    J["leg_width"] = 130
    J["sole_y"] = PEP_LEG_J["sole"] - 60  # body (crop) coords
    J["hem_y"] = PEP_HEM - 60
    json.dump(J, open(f"{out}/parts.json", "w"), indent=1)
    body = cv2.imread(f"{out}/body.png", cv2.IMREAD_UNCHANGED)
    body[PEP_HEM - 60 + 4:, :, 3] = 0
    cv2.imwrite(f"{out}/body_walk.png", body)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "cut"):
        pepper()
        coriander()
    if what in ("all", "parts"):
        pepper_parts()
    if what in ("all", "parts", "legs"):
        pepper_legs()
