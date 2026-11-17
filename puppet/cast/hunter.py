"""Distor as the hunter: his full drawing (cast/distor.py Singer: walking legs, face, lip sync) with
a 3D rifle (puppet/prop3d.py) in both hands instead of the mic.

  Hunter(song, scale, cx, top, walk=None, poses=[(t, "carry"), (t, "aim"), (t, "god")],
         shots=[t, ...], sing=[(t0, t1), ...], god=lambda t: 0..1)

poses (a schedule; hands ease between them):
- "carry": the rifle across his chest, muzzle up to the right (side view of the rifle)
- "aim": the rifle on his right shoulder, pointed at us (the rifle's "aim" view); shots kick it up
  with a muzzle flash and a puff of smoke
- "god": no rifle; his arms spread wide and up, hands open
sing: when his mouth follows the vocal track (else it stays shut). god(t): how far he is gone
(glowing eyes, a dark aura in wide framings); a number or a function of t.
"""
import math

import cv2
import numpy as np

import hand as handrig
from cast.distor import Singer
from motion import Spring
from warp import W, H, over_sub, over

GRIP = np.array([0.0, 0.44, -0.115])  # where a held stick sits in the fist (hand space, cast/distor.py mic)


class Hunter(Singer):
    LEN = 1.75  # the rifle's length in shoulder widths (his arms are short: a cartoon rifle)
    def __init__(self, song=None, scale=0.5, cx=W / 2, top=60, walk=None, poses=((0.0, "carry"),), shots=(), sing=(), god=0.0,
                 rifle="rifle", fluid=True, head=0.8, flash=1.0, aura=None):
        super().__init__(song, scale=scale, cx=cx, top=top, walk=walk, fluid=fluid, head=head, pump_after=1e9)
        from prop3d import Prop3D
        self.rifle = Prop3D(rifle)
        self.poses = sorted(poses)
        self.shots = np.array(sorted(shots), float)
        self.god = god if callable(god) else (lambda t, g=float(god): g)
        self.flash_k = flash
        self.aura = (scale < 0.7) if aura is None else aura  # the dark aura reads in wide framings; close up it would cover the frame
        n = len(self.env)
        mask = np.zeros(n, np.float32)
        for a, b in sing:
            mask[int(a * 30):int(b * 30)] = 1.0
        mask = np.convolve(mask, np.ones(5) / 5, "same")
        self.env = self.env * mask
        self.sing = self.sing * (mask > 0.5)
        self.hand_s = {"l": None, "r": None}
        self.mid = self.mic_a0  # held stick axis on screen at yaw 0 (the mic's)

    # ---- the schedule
    def pose_at(self, t):
        cur = [p for p in self.poses if p[0] <= t] or self.poses[:1]
        return cur[-1][1], t - cur[-1][0]

    def kick(self, t):
        """Recoil 0..1 after the last shot (sharp up, slower settle)."""
        if not len(self.shots):
            return 0.0, None
        i = int(np.searchsorted(self.shots, t, side="right")) - 1
        if i < 0:
            return 0.0, None
        u = t - self.shots[i]
        return (math.exp(-u / 0.09) if u < 0.6 else 0.0), u

    def look(self, t, turn, tilt):
        pose, since = self.pose_at(t)
        if pose == "aim":  # looks along the barrel, up to the right
            k = min(1.0, since / 0.3)
            return turn * (1 - 0.8 * k) + 0.35 * k, tilt * (1 - k) - 6 * k
        if pose == "god":
            return turn * 0.4, tilt * 0.5 - 3
        return turn, tilt

    # ---- arms, hands, rifle
    def _hand(self, frame, side, target, axis_deg, size, pose=None, held=True):
        """A fist around a stick at screen `target` whose axis points along axis_deg (screen angle,
        y down); returns the wrist point the arm should reach."""
        h = self.mic_hand if side == "l" else self.free_hand
        yaw = math.degrees(math.radians(axis_deg) - self.mic_a0)
        R = handrig.orient(yaw, 0, 180) if held else handrig.orient(axis_deg - 90, 10, 0)
        Mr = np.diag([-1.0, 1, 1]) if h.mirror else np.eye(3)
        T = R @ Mr
        anchor = GRIP if held else np.array(self.anchor)
        wrist = np.asarray(target, float) + ((T @ np.array(self.anchor))[:2] - (T @ anchor)[:2]) * size
        img, xy, frac = h.render(pose or handrig.POSES["fist"], R, size, target, anchor=anchor)
        return wrist, (img, xy, frac)

    def _arm(self, frame, side, sh, wrist, s):
        L = self.arm_l if side == "l" else self.arm_r
        E, Wr = L.solve(sh, wrist, s)
        lay, (x, y) = L.render(sh, E, Wr, s, (W, H))
        return lay, x, y

    def draw_arms(self, frame, t, c):
        s = self.s
        shL, shR = c["shL"], c["shR"]
        cen = (shL + shR) / 2
        sw = float(np.hypot(*(shR - shL)))
        down = np.array([0.0, 1.0])
        pose, since = self.pose_at(t)
        kick, u_shot = self.kick(t)
        hs = self.hand_size
        sway = 0.0
        if self.path is not None:
            sway = 3.0 * c["speed"] * math.sin(2 * math.pi * self.path.x(t) / self.walker.stride)
        breathe = math.sin(t * 1.7)
        draws = []
        muzzle = None
        if pose == "carry":
            grip = cen + np.array([-0.30 * sw, 0.98 * sw]) + down * (2 * breathe)
            elev = 52 + sway
            sc = self.LEN * sw / (117 * self.rifle.k)
            M = self.rifle.matrix(grip, "grip", elev, sc)
            fore = M[:, :2] @ self.rifle.anchors["fore"] + M[:, 2]
            ax = -elev  # screen angle of the rifle axis (y down)
            self.rifle.draw(frame, grip, "grip", elev, sc)
            tl, tr = grip, fore
            axl, axr = ax, ax
        elif pose == "aim":  # shouldered, aimed up to the right (at the trees, the birds, the sky)
            k = min(1.0, since / 0.25)
            k = k * k * (3 - 2 * k)
            elev = (8 + 16 * k) + 7 * kick
            butt = shL + np.array([0.08 * sw, 0.14 * sw]) + np.array([-0.05, 0.03]) * sw * kick
            butt = butt * k + (cen + np.array([-0.3 * sw, 0.95 * sw])) * (1 - k)
            sc = self.LEN * sw / (117 * self.rifle.k)
            self.rifle.draw(frame, butt, "butt", elev, sc)
            tl = self.rifle.point("grip", butt, "butt", elev, sc)
            tr = self.rifle.point("fore", butt, "butt", elev, sc)
            muzzle = self.rifle.point("muzzle", butt, "butt", elev, sc)
            axl = axr = -elev
        else:  # god: arms wide and up, hands open
            A = (self.arm_l.L1 + self.arm_l.L2) * s
            lift = 0.5 + 0.5 * min(1.0, since / 1.2)
            tl = shL + A * 0.95 * np.array([-math.sin(math.radians(60 + 65 * lift)), math.cos(math.radians(60 + 65 * lift))])
            tr = shR + A * 0.95 * np.array([math.sin(math.radians(60 + 65 * lift)), math.cos(math.radians(60 + 65 * lift))])
            tl = tl + np.array([0, 6 * math.sin(t * 1.1)])
            tr = tr + np.array([0, 6 * math.sin(t * 1.1 + 1)])
            for side, sh, tg in (("l", shL, tl), ("r", shR, tr)):
                L = self.arm_l if side == "l" else self.arm_r
                E, Wr = L.solve(sh, tg, s)
                fa = math.degrees(math.atan2(Wr[1] - E[1], Wr[0] - E[0])) - 90
                h = self.mic_hand if side == "l" else self.free_hand
                img, xy, frac = h.render(handrig.Pose.mix(handrig.POSES["relaxed"], handrig.POSES["fist"], 0.0), handrig.orient(fa, 10, 0), hs, Wr, anchor=self.anchor)
                over_sub(frame, img, xy, frac)
                lay, (x, y) = L.render(sh, E, Wr, s, (W, H))
                if lay is not None:
                    over(frame, lay, x, y)
            return
        # springs keep the hands from jumping when the pose changes
        for side, tg in (("l", tl), ("r", tr)):
            if self.hand_s[side] is None or abs(t - getattr(self, "_t_prev", t)) > 0.2:
                self.hand_s[side] = Spring(np.asarray(tg, float), omega=14.0, zeta=0.9)
        tl = self.hand_s["l"].at(np.asarray(tl, float), t)
        tr = self.hand_s["r"].at(np.asarray(tr, float), t)
        self._t_prev = t
        for side, sh, tg, ax in (("r", shR, tr, axr), ("l", shL, tl, axl)):
            wrist, hand = self._hand(frame, side, tg, ax, hs)
            draws.append((side, sh, wrist, hand))
        for side, sh, wrist, (img, xy, frac) in draws:
            over_sub(frame, img, xy, frac)
            lay, x, y = self._arm(frame, side, sh, wrist, s)
            if lay is not None:
                over(frame, lay, x, y)
        if muzzle is not None and u_shot is not None and u_shot < 0.6:
            self.muzzle_flash(frame, muzzle, u_shot, sw)

    def muzzle_flash(self, frame, p, u, sw):
        if u < 0.09:  # the flash: a star with a hot core
            k = (1 - u / 0.09) * self.flash_k
            r = 0.5 * sw * (0.6 + 0.4 * k)
            lay = np.zeros((H, W), np.float32)
            n = 9
            pts = []
            rng = np.random.default_rng(int(p[0]) + int(u * 1000))
            for i in range(2 * n):
                a = math.pi * i / n + 0.2
                rr = r * (1.0 if i % 2 == 0 else 0.32) * rng.uniform(0.7, 1.15)
                pts.append([p[0] + rr * math.cos(a), p[1] + rr * math.sin(a)])
            cv2.fillPoly(lay, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
            glow = cv2.GaussianBlur(lay, (0, 0), r * 0.25) * 1.5
            frame[:] = frame * (1 - 0.8 * k * lay[..., None]) + (lay * k)[..., None] * np.array([0.25, 0.8, 1.0], np.float32) + (glow * k)[..., None] * np.array([0.0, 0.3, 1.0], np.float32)
            np.minimum(frame, 1.0, out=frame)
        if 0.04 < u < 0.6:  # a puff of smoke drifting up
            v = (u - 0.04) / 0.56
            c = (int(p[0] + 20 * v), int(p[1] - 60 * v))
            r = int(0.25 * sw + 0.5 * sw * v)
            lay = np.zeros((H, W), np.float32)
            cv2.circle(lay, c, max(2, r), 1.0, -1, cv2.LINE_AA)
            a = cv2.GaussianBlur(lay, (0, 0), max(1.0, r * 0.4))[..., None] * 0.55 * (1 - v)
            frame[:] = frame * (1 - a) + a * np.array([0.78, 0.8, 0.82], np.float32)

    def draw(self, frame, t):
        g = float(self.god(t))
        if g > 0.01 and self.aura:  # a dark aura behind him
            self._aura(frame, t, g)
        super().draw(frame, t)
        if g > 0.01:
            self._eyes(frame, t, g)
            import story
            story.emit(lambda fr, t=t, g=g: self._eyes(fr, t, g))

    def _aura(self, frame, t, g):
        cx = self.x0 + self.wl / 2
        cy = self.top + 0.3 * self.wl
        lay = np.zeros((H // 4, W // 4), np.float32)
        r = int(self.wl * 0.42 / 4 * (1 + 0.04 * math.sin(t * 3)))
        cv2.ellipse(lay, (int(cx / 4), int(cy / 4)), (r, int(r * 1.5)), 0, 0, 360, 1.0, -1)
        lay = cv2.resize(cv2.GaussianBlur(lay, (0, 0), r * 0.45 + 1), (W, H))
        lay = np.clip((lay - 0.03) / 0.97, 0, 1)[..., None]  # no faint tails over the whole frame (venues tell the cast by what changed)
        frame[:] = frame * (1 - 0.6 * g * lay) + g * lay * np.array([0.02, 0.0, 0.18], np.float32)

    def _eyes(self, frame, t, g):
        m = self.trace.get("mouth")
        if m is None:
            return
        p = self.p
        mx, my = p.mouth[0], p.mouth[1]
        lay = np.zeros((H, W), np.float32)
        fl = 0.85 + 0.15 * math.sin(t * 17) * math.sin(t * 7)
        for ex, ey, rx, ry in p.eyes:
            c = (int(m[0] + ex - mx), int(m[1] + ey - my))
            cv2.ellipse(lay, c, (max(2, int(rx * 0.95)), max(2, int(ry * 0.8))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(lay, (0, 0), max(3.0, p.eyes[0][2] * 1.2)) * 3.0
        k = g * fl
        frame[:] = frame * (1 - k * lay[..., None]) + (lay * k)[..., None] * np.array([0.82, 0.97, 1.0], np.float32)
        frame += (glow * k)[..., None] * np.array([0.10, 0.45, 1.0], np.float32)
        np.minimum(frame, 1.0, out=frame)
