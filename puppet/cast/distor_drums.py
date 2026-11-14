"""Distor on drums: Ben's drawing of Distor (public/parts/distor) behind the drum kit of
cast/pepper.py. The kit is laid out in kit units around the shoulders; Distor's drawing is scaled
so his arms are as long, in kit units, as Pepper's (so the kit is in reach).

  DistorDrummer(song, scale=0.6, cx=W / 2, top=20, head=1.0, face=False, size=1.0)

size: how big he is next to the kit (1 = arms as long as Pepper's)."""
import json

import numpy as np

import hand as handrig
import warp
from cast.distor import arms, body_cfg
from cast.pepper import Drummer

ROOT = warp.ROOT
W, H = warp.W, warp.H
PEPPER_ARM = 450.0  # Pepper's arm (shoulder -> wrist), kit units


class DistorDrummer(Drummer):
    PARTS = f"{ROOT}/public/parts/distor"
    upper_front = True

    def __init__(self, *a, size=1.0, **kw):
        self.size = size
        super().__init__(*a, **kw)

    def rig(self, scale, cx, top):
        j = json.load(open(f"{self.PARTS}/parts.json"))
        al, ar = arms(j, self.PARTS, fold_shadow=0.2)
        arm_len = al.L1 + al.L2
        sp = scale * PEPPER_ARM / arm_len * self.size  # screen px per drawing px
        sl, sr = (np.array(j["shoulders"][k], float) for k in ("left", "right"))
        mid = (sl + sr) / 2
        # kit unit (403, 650) = Pepper's shoulder line; put his shoulder line there, a little lower
        # (his jacket shoulders are high and square)
        self.kit_anchor = (cx, top + 650 * scale)
        ax, ay = self.kit_anchor
        self.x0 = int(ax - mid[0] * sp)
        self.top_body = ay - mid[1] * sp - 40 * scale
        cfg = body_cfg(j, "body_nolegs.png", sp, (H - self.top_body) / sp + 60)
        self.p = warp.Puppet(cfg)
        self.cfg, self.j, self.sp = cfg, j, sp
        self.arms = {"l": al, "r": ar}
        self.shoulders = {"l": sl, "r": sr}
        hs = j["hand"]
        ben = handrig.Style(skin=np.array(hs["skin_bgr"]) / 255, ink=(0.075, 0.07, 0.08), ink_w=0.03, shade=0.0, flat=True)
        self.hands = {"l": handrig.StickHand(hs["skin_bgr"], hs["length"] * sp, mirror=False, style=ben),
                      "r": handrig.StickHand(hs["skin_bgr"], hs["length"] * sp, mirror=True, style=ben)}
        self.hand_anchor = (0.0, 0.10, 0.0)

    def head_args(self, e):
        return {}  # Ben's drawing: no smile/brow warps (his face reads through the eyes and blinks)
