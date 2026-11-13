"""Distor, the singer: Ben's drawing (public/parts/distor), mic in a rigged hand, his drawn
mouth and closed eyes, timed to the vocal track."""
import json
import math

import cv2
import numpy as np

import hand as handrig
import warp
from limb import Limb, join_parts
from motion import HeadMotion, Spring
from song import Song
from warp import load_rgba, over_sub

ROOT = warp.ROOT
FPS, W, H = warp.FPS, warp.W, warp.H


class Singer:
    PARTS = f"{ROOT}/public/parts/distor"

    def __init__(self, song=None, scale=0.5, cx=W / 2, top=60):
        self.song = song or Song.default()
        j = json.load(open(f"{self.PARTS}/parts.json"))
        pad = j["pad"]
        base = warp.CHARS["distor"]
        sx = lambda v: v + pad  # noqa: E731
        cfg = dict(base)
        cfg.update(full="public/parts/distor/body.png", src_h=1600, height=1600 * scale, crop_h_src=(H - top) / scale + 120,
                   head_ells=[(sx(x), y, rx, ry) for x, y, rx, ry in base["head_ells"]],
                   neck=(sx(base["neck"][0]), sx(base["neck"][1]), base["neck"][2], base["neck"][3]),
                   face=(sx(base["face"][0]),) + tuple(base["face"][1:]), pivot=(sx(base["pivot"][0]), base["pivot"][1]),
                   mouth=(sx(base["mouth"][0]),) + tuple(base["mouth"][1:]),
                   eyes=[(sx(x), y, rx, ry) for x, y, rx, ry in base["eyes"]], lid_color=(sx(base["lid_color"][0]), base["lid_color"][1]),
                   irises=[(sx(x), y, r) for x, y, r in base.get("irises", [])],
                   eye_hide=[(sx(x), y, rx, ry) for x, y, rx, ry in base.get("eye_hide", [])])
        self.p = warp.Puppet(cfg)
        self.s = scale
        self.cfg, self.j = cfg, j
        hl, wl = self.p.head.shape[:2]
        self.x0, self.top = int(cx - wl / 2), top
        up = load_rgba(f"{self.PARTS}/{j['upper_arm']['file']}")
        fo = load_rgba(f"{self.PARTS}/{j['forearm']['file']}")
        u, f, jt = join_parts(up, fo, j["upper_arm"], j["forearm"])
        self.arm_l = Limb(u, f, jt, j["arm_width"], inner=-1)  # viewer's left arm, bends toward the body
        uf, ff = u[:, ::-1].copy(), f[:, ::-1].copy()  # mirrored copy for the other arm
        wj = u.shape[1] - 1
        jm = {k: [wj - v[0], v[1]] for k, v in jt.items()}
        self.arm_r = Limb(uf, ff, jm, j["arm_width"], inner=1)
        hs = j["hand"]
        ben = handrig.Style(skin=np.array(hs["skin_bgr"]) / 255, ink=(0.075, 0.07, 0.08), ink_w=0.03, shade=0.0, flat=True)
        self.mic_hand = handrig.Hand(ben, mirror=False)  # his right hand (on the viewer's left)
        self.free_hand = handrig.Hand(ben, mirror=True)
        self.hand_size = hs["length"] * scale
        g, ax = np.array([0.0, 0.44, -0.115]), np.array([1.0, 0, 0])
        self.mic_head = g + 0.56 * ax
        self.mic = [dict(p0=g - 0.42 * ax, p1=g + 0.32 * ax, r0=0.050, r1=0.068, color=(0.13, 0.13, 0.15), gloss=0.5),
                    dict(p0=g + 0.36 * ax, p1=g + 0.38 * ax, r0=0.090, r1=0.090, color=(0.75, 0.75, 0.78), gloss=0.6),
                    dict(p0=self.mic_head, p1=self.mic_head, r0=0.165, r1=0.165, color=(0.62, 0.63, 0.66), mesh=True, gloss=0.5)]
        R0 = handrig.orient(0, 0, 180)
        d0, h0 = R0 @ ax, R0 @ np.array([0, 1.0, 0])
        self.mic_a0 = math.atan2(d0[1], d0[0])  # mic axis on screen at yaw 0
        self.hand_a0 = math.atan2(h0[1], h0[0])  # wrist-to-fingers axis on screen at yaw 0
        self.anchor = (0.0, -0.05, 0.0)
        self.env, self.vowels, self.sing = self.song.vocal_drivers()
        self.motion = HeadMotion(1, self.song.vocal_accents(), self.song.duration, self.song.spb, turn_amp=0.5, tilt_amp=3.0, nod_px=6 * scale / 0.5)
        self.up = Spring(0.0, omega=5.0)
        self.free = None
        self.prev_turn = 0.0

    def advance(self, f):
        """Keep the mic arm's spring running while off camera (called every frame by Production)."""
        for g in range(getattr(self, "_f", 0), f):
            self.up.step(self.sing[min(g, len(self.sing) - 1)])
        self._f = f + 1

    def world(self, q, y_off=0.0):
        return np.array([self.x0 + q[0] * self.s, self.top + y_off + q[1] * self.s])

    def draw(self, frame, t):
        p, cfg = self.p, self.cfg
        f = min(int(t * FPS), len(self.env) - 1)
        ph = (t % self.song.spb) / self.song.spb
        on_beat = math.cos(math.pi * ph) ** 2
        singing = self.sing[f]
        turn, tilt, nod, blink, sway = self.motion.at(t)
        turn *= 0.7 if singing else 1.0  # sings mostly toward the mic
        sway = sway * 40 * p.s + 3 * p.s * math.sin(t * 2.3)
        fig = p.head_frame(turn, float(self.env[f]), self.vowels[f], blink, sway)
        th = math.radians(tilt) * p.w_bend
        px, py = p.pivot
        ox, oy = p.X - px, p.Y - py
        c, sn = np.cos(th), np.sin(th)
        mapx = px + c * ox + sn * oy
        mapy = py - sn * ox + c * oy - nod * p.w_bend
        fig = cv2.remap(fig, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bob = 2.5 * math.sin(t * 1.7) + 0.35 * nod  # breathing, and the shoulders follow a nod a little
        warp.over(frame, fig, self.x0, int(self.top + bob))
        # where the mouth is now (after turn, tilt and nod)
        mx, my, mw, J = p.mouth
        q = np.array([mx + turn * cfg["turn_px"] * p.s * 2, my + 0.5 * J * self.env[f]])
        a = math.radians(tilt)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        mouth = np.array([self.x0, self.top + bob]) + np.array([px, py]) + R @ (q + [0, nod] - [px, py])
        s = self.s
        shL = self.world(self.j["shoulders"]["left"], bob)
        shR = self.world(self.j["shoulders"]["right"], bob)
        # the free arm (viewer's right): hangs and swings; in the chorus it pumps on beats 1 and 3
        chorus = t >= 20.0
        beat_i = int(t / self.song.spb) % 4
        pump = chorus and singing and beat_i in (0, 2) and ph < 0.5
        free_t = shR + np.array([-60, 980]) * s + np.array([12 * math.sin(t * 2.0), 0]) * s
        if pump:
            free_t = shR + np.array([-180, 420 - 160 * math.sin(math.pi * ph * 2)]) * s
        if self.free is None:
            self.free = Spring(free_t, omega=6.0, zeta=0.95)
        wt = self.free.step(free_t)
        E, Wr = self.arm_r.solve(shR, wt, s)
        fa = math.degrees(math.atan2(Wr[1] - E[1], Wr[0] - E[0])) - 90
        img, xy, frac = self.free_hand.render(handrig.POSES["fist" if pump else "relaxed"], handrig.orient(fa, 10, 0), self.hand_size, Wr, anchor=self.anchor)
        over_sub(frame, img, xy, frac)
        lay, (x, y) = self.arm_r.render(shR, E, Wr, s, (W, H))
        if lay is not None:
            warp.over(frame, lay, x, y)
        # the mic arm: up to the mouth while singing, down at the chest in pauses
        u = float(np.clip(self.up.step(self.sing[f]), 0, 1))
        u = u * u * (3 - 2 * u)
        head_up = mouth + np.array([-0.10 * mw, 1.45 * mw])
        head_dn = mouth + np.array([-2.0 * mw, 5.0 * mw])
        head = head_dn * (1 - u) + head_up * u
        phi = math.radians(-90 + (-5 * (1 - u) + 22 * u) + 4 * math.sin(t * 1.3))  # mic axis on screen (up = -90)
        L = self.arm_l
        fore = getattr(self, "_fore", -math.pi / 4)
        Mr = np.eye(3)
        ax = np.array([1.0, 0, 0])
        for _ in range(4):
            # the hand follows the forearm (wrist bend up to 45 deg); the rest turns the mic in the fist
            yaw_mic = math.degrees(phi - self.mic_a0)
            yaw_arm = math.degrees(fore - self.hand_a0)
            dy = (yaw_mic - yaw_arm + 180) % 360 - 180
            yaw = yaw_arm + max(-45, min(45, dy))
            R = handrig.orient(yaw, 0, 180)
            T = R @ Mr
            delta = phi - (math.radians(yaw) + self.mic_a0)
            rot = np.linalg.solve(T, handrig._rot((0, 0, 1), delta) @ T)  # in-hand turn of the mic
            g = np.array([0.0, 0.44, -0.115])
            head3 = g + rot @ (self.mic_head - g)
            off = ((T @ head3)[:2] - (T @ np.array(self.anchor))[:2]) * self.hand_size
            E, Wr = L.solve(shL, head - off, s)
            fore = math.atan2(Wr[1] - E[1], Wr[0] - E[0])
        self._fore = fore
        held = [dict(q, p0=g + rot @ (np.array(q["p0"]) - g), p1=g + rot @ (np.array(q["p1"]) - g)) for q in self.mic]
        img, xy, frac = self.mic_hand.render(handrig.POSES["mic"], R, self.hand_size, Wr, anchor=self.anchor, held=held)
        over_sub(frame, img, xy, frac)
        lay, (x, y) = L.render(shL, E, Wr, s, (W, H))
        if lay is not None:
            warp.over(frame, lay, x, y)


