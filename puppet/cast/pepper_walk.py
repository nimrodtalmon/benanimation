"""Pepper walking (David Revoy's model sheet, CC BY 4.0): the drummer's body and arms, legs cut
from the sheet (puppet/cutout_paint.py legs), a drumstick in each hand. Uses the shared walking
legs (puppet/legs.py) and the fluid-motion helpers.

  PepperWalk(song, scale, path, floor_y)   # path: motion.Path of her center x
"""
import json
import math

import cv2
import numpy as np

import hand as handrig
import props
import warp
from cast.pepper import PEPPER_HEAD
from legs import Legs
from limb import Limb
from motion import ArmFollow, Drag, HeadMotion, Lean
from song import Song
from warp import load_rgba, over_sub

ROOT = warp.ROOT
FPS, W, H = warp.FPS, warp.W, warp.H
MID = 403  # her middle (body crop x)


class PepperWalk:
    PARTS = f"{ROOT}/public/parts/pepper"

    def __init__(self, song=None, scale=0.4, path=None, floor_y=1000.0, look=0.55):
        self.song = song or Song.default()
        j = json.load(open(f"{self.PARTS}/parts.json"))
        self.j, self.s, self.path, self.floor_y, self.look = j, scale, path, floor_y, look
        cfg = dict(PEPPER_HEAD, full="public/parts/pepper/body_walk.png", height=1400 * scale)
        self.p = warp.Puppet(cfg)
        self.legs = Legs(self.PARTS, j["legs"], j["leg_width"], scale, path)
        self.arms = {}
        for side, inner in (("l", -1), ("r", 1)):
            a = j[f"arm_{side}"]
            self.arms[side] = Limb(load_rgba(f"{self.PARTS}/{a['file']}"), load_rgba(f"{self.PARTS}/{a['file']}"), a, j["arm_width"], inner=inner, fold_shadow=0.2)
        hs = j["hand"]
        self.hands = {"l": handrig.StickHand(hs["skin_bgr"], hs["length"] * scale, mirror=False),
                      "r": handrig.StickHand(hs["skin_bgr"], hs["length"] * scale, mirror=True)}
        self.anchor = (0.0, -0.04, 0.0)
        self.follow = {"l": ArmFollow(bend=-1.0, w_sh=7.0, z_sh=0.8, w_el=5.5, z_el=0.6), "r": ArmFollow(bend=1.0, w_sh=7.0, z_sh=0.8, w_el=5.5, z_el=0.6)}
        self.drag = {"l": Drag(), "r": Drag()}
        self.lean = Lean(max_deg=4.0)
        self.motion = HeadMotion(11, [], self.song.duration, self.song.spb, turn_amp=0.3, tilt_amp=2.0, nod_px=4 * scale)
        self.trace = {}

    def world(self, q, x0, top):
        return np.array([x0 + q[0] * self.s, top + q[1] * self.s])

    def draw(self, frame, t):
        s, p, j = self.s, self.p, self.j
        gb = self.legs.body(t)
        cx = self.path.x(t)
        x0 = cx - MID * s
        top = self.floor_y - j["sole_y"] * s + gb["bob_px"]
        # body pose: a small tilt over the stance hip and a lean into the walk, about the pelvis
        lean = self.lean.at(gb["v"] / max(s, 1e-3) * 0.245 * 0.6, t)
        piv = self.world((MID, 1050), x0, top)
        rot = gb["shoulder_tilt"] + lean
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        shift = np.array([gb["sway_px"] * 0.6, 0.0])
        X = lambda q: piv + R @ (self.world(q, x0, top) - piv) + shift  # noqa: E731
        # head: looks where she walks, settles toward us when she stops
        turn, tilt, nod, blink, sway = self.motion.at(t)
        turn = float(turn + self.look * gb["dir"] * gb["speed"])
        tilt -= 0.7 * math.degrees(rot)
        fig = p.head_frame(turn, 0.0, "ah", blink, sway * 30 * p.s)
        th = math.radians(tilt) * p.w_bend
        px, py = p.pivot
        ox, oy = p.X - px, p.Y - py
        c, sn = np.cos(th), np.sin(th)
        fig = cv2.remap(fig, (px + c * ox + sn * oy).astype(np.float32), (py - sn * ox + c * oy - nod * p.w_bend).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # legs behind the skirt
        hips = {k: X(self.legs.hip_body[k]) for k in ("l", "r")}
        rest_floor = self.floor_y + gb["bob_px"] * 0.0
        self.legs.draw(frame, t, hips, rest_floor, cx)
        o = np.array([x0, top], float)
        M = np.hstack([R, (R @ (o - piv) + piv + shift)[:, None]])
        props.blit(frame, fig, M)
        # arms hang and swing against the legs (overlap: shoulder leads, elbow trails)
        feet, _, _ = self.legs.walk.feet(t)
        for side in ("l", "r"):
            a = j[f"arm_{side}"]
            L = self.arms[side]
            S = X(a["shoulder"])
            rest = X(a["wrist"]) - S
            ln = np.hypot(*rest)
            same = feet[side][0] - cx  # the arm goes forward as its own-side leg goes back
            sw = float(np.clip(-same / (0.5 * self.legs.walk.stride), -1, 1)) * gb["speed"]
            ang = math.radians(11) * sw * gb["dir"]
            c_, s_ = math.cos(ang), math.sin(ang)
            tgt = S + np.array([c_ * rest[0] - s_ * rest[1], s_ * rest[0] + c_ * rest[1]]) * 0.97
            E, Wr, a2, om = self.follow[side].at(S, tgt, L.L1 * s, L.L2 * s, t)
            dr = self.drag[side].at(om, t)
            sg = -1 if side == "l" else 1
            hd = self.hands[side]
            yaw, s_hand = hd.solve(a2 + dr + sg * 0.35, a2)
            img, xy, frac = hd.render(yaw, s_hand, 0.0, Wr, self.anchor)
            lay, (x, y) = L.render(S, E, Wr, s, (W, H))
            if lay is not None:
                warp.over(frame, lay, x, y)
            over_sub(frame, img, xy, frac)
            self.trace.update({f"hand_{side}": Wr, f"elbow_{side}": E})
        self.trace.update(hip_l=hips["l"], hip_r=hips["r"], head=X((MID, 420)))
