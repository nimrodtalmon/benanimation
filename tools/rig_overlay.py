"""Draw a rig's head landmarks and shoulders on its drawing, to check them before animating (the mouth line must sit
on the drawn lips, the chin at the real chin): python3 tools/rig_overlay.py <rig> -> out/rig_<rig>.jpg"""
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1]
r = json.load(open(f"{ROOT}/public/rigs/{name}.json"))
img = cv2.imread(f"{ROOT}/{r['src']}")
h = r.get("head", {})
P = lambda *v: tuple(int(round(x)) for x in v)  # noqa: E731
for cx, cy, rx, ry in h.get("head_ells", []):
    cv2.ellipse(img, P(cx, cy), P(rx, ry), 0, 0, 360, (200, 200, 0), 3)
if "face" in h:
    cx, cy, rx, ry = h["face"]
    cv2.ellipse(img, P(cx, cy), P(rx, ry), 0, 0, 360, (0, 200, 0), 3)
for cx, cy, rx, ry in h.get("eyes", []):
    cv2.ellipse(img, P(cx, cy), P(rx, ry), 0, 0, 360, (255, 0, 0), 3)
if "mouth" in h:
    mx, my, hw, drop = h["mouth"]
    cv2.line(img, P(mx - hw, my), P(mx + hw, my), (0, 0, 255), 4)
    cv2.line(img, P(mx, my), P(mx, my + drop), (0, 0, 255), 2)
if "chin" in h:
    cv2.line(img, P(0, h["chin"]), P(img.shape[1], h["chin"]), (0, 140, 255), 2)
if "pivot" in h:
    cv2.circle(img, P(*h["pivot"]), 12, (255, 0, 255), -1)
for side, (x, y) in r.get("shoulders", {}).items():
    cv2.circle(img, P(x, y), 16, (0, 255, 255), -1)
os.makedirs(f"{ROOT}/out", exist_ok=True)
cv2.imwrite(f"{ROOT}/out/rig_{name}.jpg", img)
print(f"out/rig_{name}.jpg")
