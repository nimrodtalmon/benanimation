"""A head-and-shoulders drawing as a shot of its own (side views, close-ups): the head drifts
like a player's (Groove), the face reacts to the music, and it sings if it is the lead voice.
Any rig of kind "portrait" (public/rigs/*.json -> puppet/build_bust.py).

  Portrait(song, "blonde_side", head_px=620, head_top=90, cx=W / 2, head=1.0, sings=False)
"""
import math

import cv2
import numpy as np

import rig
import warp
from groove import Groove
from motion import Face, decay_since

W, H = warp.W, warp.H


class Portrait:
    def __init__(self, song, name, head_px=620, head_top=90, cx=W / 2, head=1.0, seed=0, style="mix", shouts=(), sings=False):
        self.song, self.name = song, name
        j = rig.load(name)
        self.j = j
        fx, fy, frx, fry = j["head"]["face"][:4]
        s = head_px / (2 * fry)
        top = head_top - (fy - fry) * s
        self.s, self.top = s, top
        cfg = rig.puppet_cfg(name, j, s, (H - top) / s + 80, src_h=1000)
        self.p = warp.Puppet(cfg)
        self.x0 = int(cx - (fx + j.get("pad", 0)) * s)
        self.groove = Groove(song, seed=seed + 3, amount=head, style=style)
        crashes = [d["t"] for d in song.data.get("drums", []) if d["piece"] == 49]
        self.face = Face(song, accents=crashes, hits=[], base_smile=0.05, seed=seed)
        self.shouts = np.sort(np.asarray(shouts, float))
        self.blinks = j.get("blink", True)
        self.sings = sings
        self.voice = song.vocal_drivers() if sings else None

    def credit(self):
        return self.j["credit"]

    def draw(self, frame, t):
        p, s = self.p, self.s
        g = self.groove.at(t)
        turn, tilt, nod = g.get("turn", 0.0), g.get("tilt", 0.0), g["nod"] * s / 0.5
        e = self.face.at(t)
        shout, _ = decay_since(t, self.shouts, 0.45) if len(self.shouts) else (0.0, 0)
        mouth, vowel = min(0.85, e["mouth"] + 0.75 * shout), "ah"
        smile = min(1.0, e["smile"] + 0.45 * g["energy"] - 0.3 * shout)
        if self.sings:
            env, vowels, _ = self.voice
            fi = min(len(env) - 1, (int(t * 30) // 2) * 2)
            if env[fi] > 0.02:
                mouth, vowel, smile = max(mouth, 0.9 * float(env[fi])), vowels[fi], smile * 0.4
        fig = p.head_frame(turn, mouth, vowel, min(1.0, e["squint"] + 0.25 * shout) if self.blinks else 0.0, 0.0,
                           smile=smile, brow=min(1.0, e["brow"] + 0.6 * shout))
        th = math.radians(tilt) * p.w_bend
        hx, hy = p.pivot
        ox, oy = p.X - hx, p.Y - hy
        c, sn = np.cos(th), np.sin(th)
        fig = cv2.remap(fig, (hx + c * ox + sn * oy).astype(np.float32), (hy - sn * ox + c * oy - nod * p.w_bend).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp.over(frame, fig, int(round(self.x0 + g["dx"] * s / 0.5)), int(round(self.top + g["dy"] * s / 0.5)))
