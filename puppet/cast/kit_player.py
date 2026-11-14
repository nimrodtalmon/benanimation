"""Any rigged character at the drum kit of cast/pepper.py (Drummer). The kit is laid out in kit
units around the shoulders; the character's drawing is scaled so its arms are as long, in kit
units, as Pepper's (so the kit is in reach), and the arm joints can sit a little inside drawn
shoulder pads. Nothing here is specific to one character: everything comes from its parts.json.

  KitPlayer(song, "distor", scale=0.6, cx=W / 2, top=20, head=1.0, face=False, size=1.0,
            shoulder_inset=0.07, sit=40)
"""
import numpy as np

import hand as handrig
import rig
import warp
from cast.pepper import Drummer

W, H = warp.W, warp.H
PEPPER_ARM = 450.0  # Pepper's arm (shoulder -> wrist), kit units


class KitPlayer(Drummer):
    upper_front = True  # sleeves over the body (drawn jackets hide the shoulder joints)

    def __init__(self, song, name, *a, size=1.0, shoulder_inset=0.07, sit=40, **kw):
        """size: how big the player is next to the kit (1 = arms as long as Pepper's);
        shoulder_inset: arm joints moved in from the drawn shoulder points (fraction of the shoulder
        width; padded shoulders sit outside the real joint); sit: how much lower than Pepper's
        the shoulder line sits (kit units)."""
        self.name, self.size, self.inset, self.sit = name, size, shoulder_inset, sit
        self.PARTS = rig.parts_dir(name)
        super().__init__(song, *a, **kw)

    def rig(self, scale, cx, top):
        j = rig.load(self.name)
        al, ar = rig.arms(self.name, j, fold_shadow=0.2)
        sp = scale * PEPPER_ARM / (al.L1 + al.L2) * self.size  # screen px per drawing px
        sl, sr = (np.array(j["shoulders"][k], float) for k in ("left", "right"))
        mid = (sl + sr) / 2
        # kit unit (403, 650) = Pepper's shoulder line; put this player's shoulder line there
        self.kit_anchor = (cx, top + 650 * scale)
        ax, ay = self.kit_anchor
        self.x0 = int(ax - mid[0] * sp)
        self.top_body = ay - mid[1] * sp - self.sit * scale
        hd = j.get("head") or warp.CHARS[self.name]
        cfg = rig.puppet_cfg(self.name, j, sp, (H - self.top_body) / sp + 60, body_file="body_nolegs.png" if "head" not in j else None,
                             src_h=hd.get("src_h", 1000))
        self.p = warp.Puppet(cfg)
        self.cfg, self.j, self.sp = cfg, j, sp
        self.arms = {"l": al, "r": ar}
        dx = self.inset * (sr[0] - sl[0])
        self.shoulders = {"l": sl + np.array([dx, 0.4 * dx]), "r": sr + np.array([-dx, 0.4 * dx])}
        hs, st = j["hand"], rig.hand_style(j)
        self.hands = {"l": handrig.StickHand(hs["skin_bgr"], hs["length"] * sp, mirror=False, style=st),
                      "r": handrig.StickHand(hs["skin_bgr"], hs["length"] * sp, mirror=True, style=st)}
        self.hand_anchor = (0.0, 0.10, 0.0)

    def head_args(self, e):
        return {}  # hand-drawn faces: no smile/brow warps (they read through the eyes and blinks)
