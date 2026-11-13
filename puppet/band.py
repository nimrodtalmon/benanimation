"""The band as warp puppets, every move timed to the song (src/song.json + the vocal track).

  python3 puppet/band.py singer [start dur]   -> out/band-singer.mp4
  python3 puppet/band.py guitarist|drummer    -> out/band-<who>.mp4
  python3 puppet/band.py video                -> out/band-video.mp4 (cuts on the bar)
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp  # noqa: E402
from limb import Limb, join_parts  # noqa: E402
import props  # noqa: E402
from props import blit, microphone, place  # noqa: E402
import hand as handrig  # noqa: E402

ROOT = warp.ROOT
FPS, W, H = warp.FPS, warp.W, warp.H
SONG = json.load(open(f"{ROOT}/src/song.json"))
SPB = 60 / SONG["bpm"]
BAR = 4 * SPB


def load_rgba(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255


class Spring:
    """Critically damped follower: smooth, slightly lagging motion toward a moving target."""

    def __init__(self, x, omega=9.0, zeta=1.0):
        self.x, self.v, self.w, self.z = np.array(x, float), np.zeros_like(np.array(x, float)), omega, zeta

    def step(self, target, dt=1 / FPS):
        a = self.w * self.w * (np.asarray(target, float) - self.x) - 2 * self.z * self.w * self.v
        self.v = self.v + a * dt
        self.x = self.x + self.v * dt
        return self.x


def song_drivers():
    """Vocal loudness + vowels for the whole song, and a 'singing' flag with look-ahead."""
    n = int(SONG["duration"] * FPS)
    env, vowels = warp.drivers(0.0, n)
    env = np.concatenate([env, np.zeros(max(0, n - len(env)))])[:n]
    look = np.array([env[max(0, i - 9):i + 18].max() for i in range(n)])  # -0.3 s .. +0.6 s
    sing = look > 0.25
    # bridge short gaps (keep the mic up when the pause is under 1.4 s)
    idx = np.nonzero(sing)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a < 1.4 * FPS:
            sing[a:b] = True
    return env, vowels, sing.astype(float)


# ---------------------------------------------------------------- the singer (Ben's Distor)
class Singer:
    PARTS = f"{ROOT}/public/parts/distor"

    def __init__(self, scale=0.5, cx=W / 2, top=60):
        j = json.load(open(f"{self.PARTS}/parts.json"))
        pad = j["pad"]
        base = warp.CHARS["distor"]
        sx = lambda v: v + pad  # noqa: E731
        cfg = dict(base)
        cfg.update(full="public/parts/distor/body.png", src_h=1600, height=1600 * scale,
                   head_ells=[(sx(x), y, rx, ry) for x, y, rx, ry in base["head_ells"]],
                   neck=(sx(base["neck"][0]), sx(base["neck"][1]), base["neck"][2], base["neck"][3]),
                   face=(sx(base["face"][0]),) + tuple(base["face"][1:]), pivot=(sx(base["pivot"][0]), base["pivot"][1]),
                   mouth=(sx(base["mouth"][0]),) + tuple(base["mouth"][1:]),
                   eyes=[(sx(x), y, rx, ry) for x, y, rx, ry in base["eyes"]], lid_color=(sx(base["lid_color"][0]), base["lid_color"][1]))
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
        self.mic_hand = handrig.Hand(ben, mirror=True)
        self.free_hand = handrig.Hand(ben, mirror=True)
        self.hand_size = hs["length"] * scale
        g, ax = np.array([0.0, 0.44, -0.115]), np.array([1.0, 0, 0])
        self.mic_head = g + 0.40 * ax
        self.mic = [dict(p0=g - 0.30 * ax, p1=g + 0.24 * ax, r0=0.040, r1=0.052, color=(0.13, 0.13, 0.15), gloss=0.5),
                    dict(p0=g + 0.28 * ax, p1=g + 0.29 * ax, r0=0.066, r1=0.066, color=(0.75, 0.75, 0.78), gloss=0.6),
                    dict(p0=self.mic_head, p1=self.mic_head, r0=0.110, r1=0.110, color=(0.62, 0.63, 0.66), mesh=True, gloss=0.5)]
        R0 = handrig.orient(0, 0, 180) @ np.diag([-1.0, 1, 1])
        d0, h0 = R0 @ ax, R0 @ np.array([0, 1.0, 0])
        self.mic_a0 = math.atan2(d0[1], d0[0])  # mic axis on screen at yaw 0
        self.hand_a0 = math.atan2(h0[1], h0[0])  # wrist-to-fingers axis on screen at yaw 0
        self.anchor = (0.0, -0.05, 0.0)
        self.env, self.vowels, self.sing = song_drivers()
        self.up = Spring(0.0, omega=7.0)
        self.free = None
        self.prev_turn = 0.0

    def world(self, q, y_off=0.0):
        return np.array([self.x0 + q[0] * self.s, self.top + y_off + q[1] * self.s])

    def draw(self, frame, t):
        p, cfg = self.p, self.cfg
        f = min(int(t * FPS), len(self.env) - 1)
        ph = (t % SPB) / SPB
        on_beat = math.cos(math.pi * ph) ** 2
        singing = self.sing[f]
        turn = 0.75 * warp.smooth_noise(t, 1) * (0.6 if singing else 1.0)
        tilt = 3.0 * warp.smooth_noise(t, 4) + 1.2 * on_beat
        nod = 7 * cfg["height"] / 1500 * on_beat
        sway = (self.prev_turn - turn) * 260 * p.s + 4 * p.s * math.sin(t * 2.3)
        self.prev_turn = turn
        k = int(t / 3.3)
        bt = t - (k * 3.3 + (k * 7 % 10) / 11)
        blink = max(0.0, 1 - abs(bt - 0.07) / 0.07) if 0 <= bt < 0.14 else 0.0
        fig = p.head_frame(turn, float(self.env[f]), self.vowels[f], blink, sway)
        th = math.radians(tilt) * p.w_bend
        px, py = p.pivot
        ox, oy = p.X - px, p.Y - py
        c, sn = np.cos(th), np.sin(th)
        mapx = px + c * ox + sn * oy
        mapy = py - sn * ox + c * oy - nod * p.w_bend
        fig = cv2.remap(fig, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bob = 3 * math.sin(t * 1.7)
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
        beat_i = int(t / SPB) % 4
        pump = chorus and singing and beat_i in (0, 2) and ph < 0.5
        free_t = shR + np.array([-60, 980]) * s + np.array([12 * math.sin(t * 2.0), 0]) * s
        if pump:
            free_t = shR + np.array([-180, 420 - 160 * math.sin(math.pi * ph * 2)]) * s
        if self.free is None:
            self.free = Spring(free_t, omega=11.0, zeta=0.75)
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
        head_up = mouth + np.array([-0.10 * mw, 1.15 * mw]) + np.array([0, 0.1 * mw * on_beat])
        head_dn = mouth + np.array([-2.0 * mw, 5.0 * mw])
        head = head_dn * (1 - u) + head_up * u
        phi = math.radians(-90 + (-5 * (1 - u) + 22 * u) + 4 * math.sin(t * 1.3))  # mic axis on screen (up = -90)
        L = self.arm_l
        fore = getattr(self, "_fore", -math.pi / 4)
        Mr = np.diag([-1.0, 1, 1])
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


# ---------------------------------------------------------------- rigged hands
def over_sub(dst, img, xy, frac):
    """Composite a BGRA layer at a sub-pixel position (smooth, delicate motion)."""
    if abs(frac[0]) > 1e-3 or abs(frac[1]) > 1e-3:
        M = np.float32([[1, 0, frac[0]], [0, 1, frac[1]]])
        img = cv2.warpAffine(img, M, (img.shape[1] + 1, img.shape[0] + 1), flags=cv2.INTER_LINEAR, borderValue=0)
    warp.over(dst, img, int(xy[0]), int(xy[1]))


class StickHand:
    """A rigged hand holding a drumstick. The stick sits between thumb and index (the fulcrum);
    the wrist bends up to a limit, the rest of the stick angle is a pivot in the fingers."""
    FULCRUM = np.array([0.06, 0.44, -0.115])
    S0 = np.array([0.75, 0.62, 0.0]) / np.linalg.norm([0.75, 0.62, 0.0])

    def __init__(self, skin, size, mirror, pitch=62, roll=18, style=None):
        st = style or handrig.Style(skin=np.array(skin) / 255, ink=(0.16, 0.12, 0.12), ink_w=0.016, shade=0.5)
        self.h = handrig.Hand(st, mirror=mirror)
        self.size, self.mirror, self.pitch, self.roll = size, mirror, pitch, roll
        self.L = 1.85  # stick length in hand lengths
        self.Mr = np.diag([-1.0, 1, 1]) if mirror else np.eye(3)

    def _T(self, yaw):
        return handrig.orient(yaw, self.pitch, self.roll) @ self.Mr

    def solve(self, phi, fore_angle):
        """phi: wanted stick angle on screen (rad); fore_angle: forearm angle (rad).
        Returns yaw (deg) and the stick direction in hand space."""
        T0 = self._T(0)
        d0 = T0 @ self.S0
        a0 = math.atan2(d0[1], d0[0])
        yaw_stick = math.degrees(phi - a0)
        yaw_arm = math.degrees(fore_angle) - 90
        dy = (yaw_stick - yaw_arm + 180) % 360 - 180
        yaw = yaw_arm + max(-38, min(38, dy))
        delta = phi - (math.radians(yaw) + a0)
        T = self._T(yaw)
        Rz = handrig._rot((0, 0, 1), delta)
        s_hand = np.linalg.solve(T, Rz @ (T @ self.S0))
        return yaw, s_hand

    def tip_offset(self, yaw, s_hand, anchor):
        T = self._T(yaw)
        tip = self.FULCRUM + s_hand * self.L * 0.78
        return ((T @ tip)[:2] - (T @ np.asarray(anchor))[:2]) * self.size

    def render(self, yaw, s_hand, loose, wrist, anchor):
        pose = handrig.Pose.mix(handrig.POSES["stick"], handrig.POSES["stick_loose"], float(np.clip(loose, 0, 1)))
        g = self.FULCRUM
        held = [dict(p0=g - s_hand * self.L * 0.22, p1=g + s_hand * self.L * 0.78, r0=0.042, r1=0.028, color=(0.42, 0.66, 0.84), gloss=0.35),
                dict(p0=g + s_hand * self.L * 0.79, p1=g + s_hand * self.L * 0.80, r0=0.036, r1=0.036, color=(0.50, 0.74, 0.90), gloss=0.5)]
        return self.h.render(pose, handrig.orient(yaw, self.pitch, self.roll), self.size, wrist, anchor=anchor, held=held)


# ---------------------------------------------------------------- the drummer (Revoy's Pepper)
PEPPER_HEAD = dict(
    src_h=1400, head_ells=[(404, 220, 330, 200), (403, 420, 150, 160)], neck=(368, 438, 505, 600),
    face=(403, 420, 100, 125), pivot=(403, 560), mouth=(403, 466, 22, 16), chin=518,
    eyes=[(355, 380, 24, 26), (452, 380, 24, 26)], lid_from=70, lash=(45, 30, 30), lash_w=7,
    hair_from=450, turn_px=10)
DRUM = dict(KICK=36, SNARE=38, HAT=42, OPEN_HAT=46, CRASH=49, TOM_HI=50, TOM_MID=47, TOM_LO=45)


def decay_since(t, times, tau):
    """exp(-(t - last event)/tau) for the most recent event before t (0 if none)."""
    i = np.searchsorted(times, t, side="right") - 1
    return math.exp(-(t - times[i]) / tau) if i >= 0 else 0.0, (t - times[i]) if i >= 0 else 9.0


class Drummer:
    PARTS = f"{ROOT}/public/parts/pepper"

    def __init__(self, scale=0.72, cx=W / 2, top=30):
        j = json.load(open(f"{self.PARTS}/parts.json"))
        cfg = dict(PEPPER_HEAD, full="public/parts/pepper/body.png", height=1400 * scale)
        self.p = warp.Puppet(cfg)
        self.cfg, self.j, self.s = cfg, j, scale
        self.x0, self.top = int(cx - 403 * scale), top
        self.arms = {}
        for side, inner in (("l", -1), ("r", 1)):
            a = j[f"arm_{side}"]
            img = load_rgba(f"{self.PARTS}/{a['file']}")
            self.arms[side] = Limb(img, img, a, j["arm_width"], inner=inner, fold_shadow=0.2)
        s = scale
        P = lambda x, y: (self.x0 + x * s, top + y * s)  # noqa: E731  crop coords -> screen
        # the kit, placed in the drummer's (crop) coordinates
        skull = cv2.imread(f"{ROOT}/public/clip/skull.png", cv2.IMREAD_UNCHANGED)
        self.kit = dict(
            kick=(props.kick_front(330 * s, skull), P(403, 1470)),
            floor=(props.drum(165 * s, 180 * s, seed=4), P(70, 1255)),
            tom_hi=(props.drum(128 * s, 105 * s, seed=1), P(300, 1100)),
            tom_mid=(props.drum(135 * s, 110 * s, seed=2), P(515, 1100)),
            snare=(props.drum(150 * s, 70 * s, chrome=True, seed=5), P(225, 1185)),
            hat=(props.cymbal(132 * s), P(625, 1150)),
            hat_low=(props.cymbal(132 * s), P(625, 1168)),
            crash=(props.cymbal(185 * s), P(705, 900)),
        )
        self.floor_y = P(0, 1760)[1]
        self.ao = None
        hs = j["hand"]
        self.hands = {"l": StickHand(hs["skin_bgr"], hs["length"] * s, mirror=False), "r": StickHand(hs["skin_bgr"], hs["length"] * s, mirror=True)}
        self.hand_anchor = (0.0, -0.04, 0.0)  # the wrist sits a little inside the cuff
        # which hand plays what: viewer's left (her right hand) = snare and toms, viewer's right = hats and crash
        piece_at = {DRUM["SNARE"]: "snare", DRUM["TOM_HI"]: "tom_hi", DRUM["TOM_MID"]: "tom_mid", DRUM["TOM_LO"]: "floor",
                    DRUM["HAT"]: "hat", DRUM["OPEN_HAT"]: "hat", DRUM["CRASH"]: "crash"}
        hand_of = {"snare": "l", "tom_hi": "l", "tom_mid": "r", "floor": "l", "hat": "r", "crash": "r"}
        self.hits = {"l": [], "r": []}
        for d in sorted(SONG["drums"], key=lambda d: d["t"]):
            if d["piece"] not in piece_at:
                continue
            what = piece_at[d["piece"]]
            hand = hand_of[what]
            if self.hits[hand] and d["t"] - self.hits[hand][-1][0] < 0.11:
                other = "l" if hand == "r" else "r"
                if not self.hits[other] or d["t"] - self.hits[other][-1][0] >= 0.11:
                    hand = other
                else:
                    continue
            self.hits[hand].append((d["t"], what, d["vel"]))
        self.times = {k: np.array([d["t"] for d in SONG["drums"] if d["piece"] in v]) for k, v in dict(
            kick=[36], snare=[38], hat=[42, 46], open=[46], crash=[49], tom_hi=[50], tom_mid=[47], floor=[45]).items()}
        self.prev_turn = 0.0
        self.prev_fore = {}

    def strike_point(self, what, side):
        img, anc = self.kit[what][0]
        x, y = self.kit[what][1]
        sg = -1 if side == "l" else 1
        if what in ("hat", "crash"):
            return np.array([x + sg * 0.3 * img.shape[1] / 2, y - 2])
        return np.array([x + sg * 0.15 * img.shape[1] / 2, y + 2])

    def hand_pose(self, side, t):
        """Stick tip path from the score: arc up between hits, fast down-stroke onto the next one."""
        hits = self.hits[side]
        times = [h[0] for h in hits]
        i = np.searchsorted(times, t, side="right") - 1
        base = {"l": math.radians(55), "r": math.radians(125)}[side]
        rest_what = "snare" if side == "l" else "hat"
        if i < 0:
            p0, w0, t0 = self.strike_point(rest_what, side), rest_what, -9.0
        else:
            t0, w0, _ = hits[i]
            p0 = self.strike_point(w0, side)
        if i + 1 < len(hits):
            t1, w1, v1 = hits[i + 1]
            p1 = self.strike_point(w1, side)
        else:
            t1, w1, v1, p1 = t0 + 9, w0, 80, p0
        gap = t1 - t0
        sgn = -1 if side == "l" else 1
        cym = lambda w: w in ("hat", "crash")  # noqa: E731
        a0 = base + sgn * (math.radians(10) if cym(w0) else 0.0)
        a1 = base + sgn * (math.radians(10) if cym(w1) else 0.0)
        if gap > 1.2:  # long pause: rest, then a lead-in of 0.5 s
            if t < t1 - 0.5:
                return p0 + np.array([0, -14]) * self.s / 0.72, a0, a0 + sgn * math.radians(12), 0.15
            t0, gap = t1 - 0.5, 0.5
        u = np.clip((t - t0) / gap, 0, 1)
        e = u * u * (3 - 2 * u)
        tip = p0 * (1 - e) + p1 * e  # where the tip would be with the stick down
        ang = a0 * (1 - e) + a1 * e
        lift = math.sin(math.pi * u ** 0.7) * min(1.0, gap / 0.45) * (0.6 + 0.4 * v1 / 127)
        ang_now = ang + sgn * math.radians(58) * lift  # the wrist flicks the stick up
        return tip, ang, ang_now, lift

    def draw(self, frame, t):
        p, s = self.p, self.s
        f = int(t * FPS)
        ph = (t % SPB) / SPB
        kick, _ = decay_since(t, self.times["kick"], 0.12)
        snare, _ = decay_since(t, self.times["snare"], 0.18)
        bounce = 6 * s / 0.72 * kick
        # head: nods on the snare, glances a little toward the busy hand
        turn = 0.5 * warp.smooth_noise(t, 7)
        tilt = 2.5 * warp.smooth_noise(t, 9) + 2.0 * snare
        nod = 10 * s * snare + 4 * s * math.cos(math.pi * ph) ** 2
        sway = (self.prev_turn - turn) * 200 * p.s
        self.prev_turn = turn
        k = int(t / 2.9)
        bt = t - (k * 2.9 + (k * 5 % 9) / 10)
        blink = max(0.0, 1 - abs(bt - 0.07) / 0.07) if 0 <= bt < 0.14 else 0.0
        fig = p.head_frame(turn, 0.0, "ah", blink, sway)
        th = math.radians(tilt) * p.w_bend
        px, py = p.pivot
        ox, oy = p.X - px, p.Y - py
        c, sn = np.cos(th), np.sin(th)
        fig = cv2.remap(fig, (px + c * ox + sn * oy).astype(np.float32), (py - sn * ox + c * oy - nod * p.w_bend).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        top = self.top + bounce
        # hands: stick tips from the score; the wrist is found from the tip (hand + stick geometry), then IK
        poses = {}
        for side in ("l", "r"):
            tip, ang, ang_now, lift = self.hand_pose(side, t)
            tip = tip + np.array([0, bounce * 0.5])
            L = self.arms[side]
            L.fore_k = 0.55 + 0.2 * lift  # forearms point at the kit (toward us); lifted they show more
            S = np.array([self.x0 + L.S0[0] * s, top + L.S0[1] * s])
            hd = self.hands[side]
            fore = self.prev_fore.get(side, L.a2_0)
            for _ in range(4):  # wrist for the stick-down pose ...
                yaw, s_hand = hd.solve(ang, fore)
                wrist = tip - hd.tip_offset(yaw, s_hand, self.hand_anchor)
                wrist = wrist + np.array([0, -45 * s * lift])  # ... raised a little on the backstroke
                E, Wr = L.solve(S, wrist, s)
                fore = math.atan2(Wr[1] - E[1], Wr[0] - E[0])
            self.prev_fore[side] = fore
            yaw, s_hand = hd.solve(ang_now, fore)
            loose = lift * 1.4 - 0.15
            poses[side] = (S, E, Wr, yaw, s_hand, loose)
        # back to front: floor shadows, stands, upper arms, body (shaded behind the kit), cymbals, drums, hands, forearms
        kx, ky = self.kit["kick"][1]
        props.floor_shadow(frame, kx, min(H - 5, ky + 300 * s), 520 * s, 60 * s, 0.55)
        sw = max(4, int(8 * s / 0.72))
        for what in ("crash", "hat"):
            x, y = self.kit[what][1]
            props.stand(frame, (x, y), min(H + 30, self.floor_y), sw)
        sx, sy = self.kit["snare"][1]
        props.stand(frame, (sx, sy + 60 * s), min(H + 30, self.floor_y), sw, tilter=False)
        for side in ("l", "r"):
            S, E, Wr = poses[side][:3]
            lay, (x, y) = self.arms[side].render(S, E, Wr, s, (W, H), part="upper")
            if lay is not None:
                warp.over(frame, lay, x, y)
        warp.over(frame, fig, self.x0, int(top))
        if self.ao is None:  # the kit shades what is right behind it
            m = np.zeros((H, W, 3), np.float32)
            for what in ("floor", "tom_hi", "tom_mid", "kick", "snare"):
                img, anc = self.kit[what][0]
                blit(m, np.dstack([np.ones_like(img[..., :3]), img[..., 3:]]), place(img, anc.get("head", anc.get("center")), self.kit[what][1], 0.0, 1.0))
            m = cv2.GaussianBlur(m[..., 0], (0, 0), 30 * s)
            self.ao = (1 - 0.45 * np.clip(m * 1.4, 0, 1))[..., None].astype(np.float32)
        frame *= self.ao
        crash, _ = decay_since(t, self.times["crash"], 0.7)
        hat, dt_hat = decay_since(t, self.times["hat"], 0.15)
        opened = decay_since(t, self.times["open"], 0.35)[0]
        for what in ("hat_low", "hat", "crash"):
            img, anc = self.kit[what][0]
            x, y = self.kit[what][1]
            if what == "crash":
                ang = math.radians(9) * crash * math.sin(2 * math.pi * 3.2 * (t - (self.times["crash"][self.times["crash"] <= t][-1] if crash else 0))) - math.radians(8)
            elif what == "hat":
                y = y - 10 * s * opened + 4 * s * hat
                ang = math.radians(3) * hat
            else:
                ang = 0.0
            blit(frame, img, place(img, anc["center"], (x, y), ang, 1.0))
        for what in ("floor", "tom_hi", "tom_mid"):
            pass
        for (tw, ax) in (("tom_hi", -60), ("tom_mid", 60)):  # tom arms out of the bass drum
            tx, ty = self.kit[tw][1]
            props.tom_mount(frame, (kx + ax * s, ky - 300 * s), (tx, ty + 80 * s), max(4, int(9 * s / 0.72)))
        for what in ("floor", "tom_hi", "tom_mid", "kick", "snare"):  # back to front
            img, anc = self.kit[what][0]
            x, y = self.kit[what][1]
            key = {"tom_hi": "tom_hi", "tom_mid": "tom_mid", "floor": "floor", "snare": "snare", "kick": "kick"}[what]
            hit, _ = decay_since(t, self.times[key], 0.1)
            sc = 1.0 + 0.025 * hit
            anchor = anc.get("head", anc.get("center"))
            blit(frame, img, place(img, anchor, (x, y + 3 * hit), 0.0, sc))
        for side in ("l", "r"):
            S, E, Wr, yaw, s_hand, loose = poses[side]
            img, xy, frac = self.hands[side].render(yaw, s_hand, loose, Wr, self.hand_anchor)
            over_sub(frame, img, xy, frac)
            lay, (x, y) = self.arms[side].render(S, E, Wr, s, (W, H), part="fore")
            if lay is not None:
                warp.over(frame, lay, x, y)


# ---------------------------------------------------------------- the guitarist (Revoy's Coriander)
CORI_HEAD = dict(
    src_h=2070, head_ells=[(690, 330, 240, 225), (760, 590, 190, 170)], neck=(715, 805, 620, 700),
    face=(765, 545, 85, 105), pivot=(760, 680), mouth=(785, 588, 18, 12), chin=622,
    eyes=[(720, 505, 22, 15), (805, 500, 18, 14)], lid_from=45, lash=(40, 25, 25), lash_w=6,
    hair_from=600, turn_px=8)
CORI_STRUM_ARM = [(375, 1060), (520, 990), (585, 1060), (560, 1180), (535, 1300), (395, 1310), (360, 1190)]
CORI_STRUM_HAND = [(408, 1125), (470, 1118), (503, 1130), (512, 1195), (520, 1240), (505, 1275), (445, 1275), (430, 1200), (405, 1170)]
CORI_FRET_HAND = [(948, 1112), (1045, 1108), (1058, 1160), (1040, 1222), (975, 1222), (948, 1170)]
CHORD_POS = {"D": 0.0, "G": 1.0, "C": 0.55}


def soft_poly(shape, pts, s, sigma):
    m = np.zeros(shape, np.float32)
    cv2.fillPoly(m, [np.round(np.array(pts) * s).astype(np.int32)], 1.0, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), max(0.5, sigma * s)) if sigma > 0 else m


class Guitarist:
    def __init__(self, scale=0.78, cx=W / 2, top=None):
        cfg = dict(CORI_HEAD, full="public/paint/coriander-full.png", height=2070 * scale)
        self.p = warp.Puppet(cfg)
        self.cfg, self.s = cfg, scale
        self.j = dict(credit='Painting: "Coriander concept art" by David Revoy (peppercarrot.com), CC BY 4.0, cut out and animated')
        self.x0 = int(cx - 760 * scale)
        self.top = int(20 - 120 * scale) if top is None else top
        sh = self.p.head.shape[:2]
        s = scale
        self.w_strum = soft_poly(sh, CORI_STRUM_ARM, s, 28)
        self.m_strum = soft_poly(sh, CORI_STRUM_HAND, s, 2)
        self.w_fret = soft_poly(sh, [(935, 1095), (1070, 1090), (1085, 1170), (1055, 1240), (960, 1240), (930, 1170)], s, 14)
        self.m_fret = soft_poly(sh, CORI_FRET_HAND, s, 2)
        self.piv_strum = np.array([560, 985]) * s
        # her painted hands (and the bottle) are replaced by rigged hands: paint them out with the coat
        img = self.p.head
        rm = np.zeros(sh, np.uint8)
        for poly in (CORI_FRET_HAND, [(400, 1105), (530, 1105), (545, 1300), (420, 1310), (390, 1200)]):
            cv2.fillPoly(rm, [np.round(np.array(poly) * s).astype(np.int32)], 1)
        rm = cv2.dilate(rm, np.ones((5, 5), np.uint8))
        bgr = np.clip(img[..., :3] * 255, 0, 255).astype(np.uint8)
        fill = cv2.inpaint(bgr, rm, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
        img[..., :3] = np.where(rm[..., None] > 0, fill, img[..., :3])
        skin = (106, 159, 196)
        st = handrig.Style(skin=np.array(skin) / 255, ink=(0.14, 0.11, 0.11), ink_w=0.017, shade=0.5)
        self.strum_hand = handrig.Hand(st, mirror=False)
        self.fret_hand = handrig.Hand(st, mirror=True)
        self.hsize = 190 * s
        self.strum_wrist = np.array([458, 1188]) * s
        self.strum_elbow = np.array([540, 1000]) * s
        self.g, self.ga = props.guitar(440 * s)
        self.g_angle = math.radians(-5)
        self.strum_at = np.array([470, 1215]) * s
        # strumming keyframes: each strum crosses the strings, starting on the far side
        keys = []
        for g in SONG["guitar"]:
            sg = 1 if g["dir"] == "down" else -1
            keys += [(g["t"] - 0.05, -sg), (g["t"] + 0.05, sg)]
        keys.sort()
        self.kt = np.array([k[0] for k in keys])
        self.kv = np.array([k[1] for k in keys], float)
        self.strum_t = np.array([g["t"] for g in SONG["guitar"]])
        self.strum_acc = np.array([1.0 if g["accent"] else 0.6 for g in SONG["guitar"]])
        self.chords = [(g["t"], g["chord"]) for g in SONG["guitar"]]
        self.fret = Spring(0.0, omega=14.0, zeta=0.8)
        self.prev_turn = 0.0

    def strum_pos(self, t):
        i = np.searchsorted(self.kt, t) - 1
        if i < 0:
            return 0.0
        if i + 1 >= len(self.kt):
            return float(self.kv[-1])
        t0, t1, v0, v1 = self.kt[i], self.kt[i + 1], self.kv[i], self.kv[i + 1]
        u = np.clip((t - t0) / max(1e-3, t1 - t0), 0, 1)
        if t1 - t0 > 0.3:  # a long gap: drift back to the middle and come in again
            return float(v0 * (1 - u) ** 2 + v1 * u ** 3)
        return float(v0 + (v1 - v0) * u * u * (3 - 2 * u))

    def draw(self, frame, t):
        p, s = self.p, self.s
        ph = (t % SPB) / SPB
        on_beat = math.cos(math.pi * ph) ** 2
        turn = 0.5 * warp.smooth_noise(t, 3)
        tilt = 3.0 * warp.smooth_noise(t, 5) + 2.0 * on_beat
        nod = 8 * s * on_beat
        sway = (self.prev_turn - turn) * 200 * p.s
        self.prev_turn = turn
        k = int(t / 3.1)
        bt = t - (k * 3.1 + (k * 3 % 7) / 8)
        blink = max(0.0, 1 - abs(bt - 0.07) / 0.07) if 0 <= bt < 0.14 else 0.0
        fig = p.head_frame(turn, 0.0, "ah", blink, sway)
        # strumming forearm: rotate about the elbow; fretting hand: slide along the neck
        th = math.radians(7.0) * self.strum_pos(t)
        ci = max(0, np.searchsorted([c[0] for c in self.chords], t + 0.08) - 1)
        fpos = float(self.fret.step(CHORD_POS.get(self.chords[ci][1], 0.0)))
        d_neck = np.array([math.cos(self.g_angle), math.sin(self.g_angle)])
        X, Y = p.X, p.Y
        px, py = self.piv_strum
        ws = self.w_strum * th
        dx = (X - px) * (np.cos(ws) - 1) - (Y - py) * np.sin(ws)
        dy = (X - px) * np.sin(ws) + (Y - py) * (np.cos(ws) - 1)
        slide = -0.12 * self.ga["neck_len"] * fpos
        dx = dx + self.w_fret * (slide * d_neck[0])
        dy = dy + self.w_fret * (slide * d_neck[1] + 95 * s)  # the cuff comes down to meet the hand under the neck
        mx, my = (X - dx).astype(np.float32), (Y - dy).astype(np.float32)
        fig = cv2.remap(fig, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # head tilt / nod
        thh = math.radians(tilt) * p.w_bend
        hx, hy = p.pivot
        ox, oy = X - hx, Y - hy
        c, sn = np.cos(thh), np.sin(thh)
        fig = cv2.remap(fig, (hx + c * ox + sn * oy).astype(np.float32), (hy - sn * ox + c * oy - nod * p.w_bend).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bob = 5 * s * on_beat
        top = self.top + bob
        org = np.array([self.x0, top])
        # guitar (moves with the body), hung on a strap; strings ring after each strum
        ring, _ = decay_since(t, self.strum_t, 0.35)
        gimg = props.strings(self.g, self.ga, ring, t)
        M = place(gimg, self.ga["strum"], org + self.strum_at, self.g_angle, 1.0)
        Lb = (self.ga["neck"][0] - self.ga["bridge"][0]) / 0.67
        ox, oy = self.ga["neck"][0] - Lb, self.ga["neck"][1]
        butt = M @ np.array([ox + 0.01 * Lb, oy, 1])
        horn = M @ np.array([ox + 0.86 * Lb, oy - 0.33 * Lb, 1])
        sw = 30 * s
        # the back strap runs from the butt end behind her right side
        props.strap(frame, [butt, butt + np.array([20, -60]) * s, org + np.array([470, 1080]) * s], sw)
        warp.over(frame, fig, self.x0, int(top))
        # the guitar casts a soft shadow on her coat
        gm = np.zeros((H // 4, W // 4), np.float32)
        Ms = M.copy()
        Ms[:, 2] += np.array([14, 20]) * s
        Ms /= 4
        gm = cv2.warpAffine(self.g[..., 3], Ms, (W // 4, H // 4), flags=cv2.INTER_LINEAR)
        gm = cv2.resize(cv2.GaussianBlur(gm, (0, 0), 3 * s + 1), (W, H))[..., None]
        figm = np.zeros((H, W), np.float32)
        fh, fw = fig.shape[:2]
        y0, x0_ = int(top), self.x0
        ya, yb = max(0, y0), min(H, y0 + fh)
        xa, xb = max(0, x0_), min(W, x0_ + fw)
        figm[ya:yb, xa:xb] = fig[ya - y0:yb - y0, xa - x0_:xb - x0_, 3]
        frame *= 1 - 0.45 * gm * figm[..., None]
        # the front strap: from the horn over her left shoulder
        shoulder = org + np.array([905, 745]) * s
        props.strap(frame, [horn, horn * 0.4 + shoulder * 0.6 + np.array([-30, 10]) * s, shoulder], sw)
        blit(frame, gimg, M)
        props.button(frame, butt, 6 * s)
        props.button(frame, horn, 6 * s)
        org = np.array([self.x0, top])
        # fretting hand: fingers arch over the neck onto the fretboard, palm and thumb behind the neck
        nk0 = M @ np.array([self.ga["neck"][0], self.ga["neck"][1], 1])
        nk1 = M @ np.array([self.ga["nut"][0], self.ga["nut"][1], 1])
        fret_at = nk0 + (nk1 - nk0) * (0.55 - 0.12 * fpos)
        g_ang = self.g_angle
        neck_w = self.ga["neck_w"]
        yaw_f = math.degrees(g_ang) + 180  # fingers point up, hand x runs along the neck
        Rf = handrig.orient(yaw_f, -30, 180)
        neck_c = np.array([0.0, 0.62, 0.0])
        held = [dict(p0=neck_c - [0.9, 0, 0], p1=neck_c + [0.9, 0, 0], r0=neck_w / 2 / self.hsize, r1=neck_w / 2 / self.hsize, hidden=True)]
        press = 0.15 * decay_since(t, self.strum_t, 0.12)[0]
        fp = handrig.POSES["fret"]
        pose = handrig.Pose({k: tuple(a * (1 + press) for a in v) for k, v in fp.curls.items()}, fp.thumb, fp.thumb_dir, fp.spread)
        img, xy, frac = self.fret_hand.render(pose, Rf, self.hsize, fret_at, anchor=neck_c, held=held)
        over_sub(frame, img, xy, frac)
        # picking hand: hangs from her (warped) cuff over the strings, pick between thumb and index
        q, pv = self.strum_wrist, self.piv_strum
        c_, s_ = math.cos(th), math.sin(th)
        wr = pv + np.array([[c_, -s_], [s_, c_]]) @ (q - pv)
        el = self.strum_elbow
        fa = math.atan2(wr[1] - el[1], wr[0] - el[0])
        Rs = handrig.orient(math.degrees(fa) - 90 - 25 + 10 * self.strum_pos(t), 25, -15)
        pick = np.array([0.20, 0.52, -0.10])
        held = [dict(p0=pick, p1=pick + [0.05, 0.05, 0], r0=0.045, r1=0.02, color=(0.25, 0.25, 0.85), gloss=0.5)]
        img, xy, frac = self.strum_hand.render(handrig.POSES["pick"], Rs, self.hsize, org + wr, anchor=(0, -0.04, 0), held=held)
        over_sub(frame, img, xy, frac)


# ---------------------------------------------------------------- stage, camera, output
def stage_light(frame, t):
    """Darken toward the floor (the light comes from above) and add a little haze."""
    if not hasattr(stage_light, "ramp"):
        yy = np.linspace(0, 1, H, dtype=np.float32)
        stage_light.ramp = (1 - 0.7 * np.clip((yy - 0.72) / 0.28, 0, 1) ** 1.3)[:, None, None]
    frame *= stage_light.ramp
    return frame


def finish(frame, t, f, n, rng, credit, zoom=(1.0, 1.06), focus=(W / 2, H * 0.42)):
    z = zoom[0] + (zoom[1] - zoom[0]) * f / max(1, n - 1)
    C = cv2.getRotationMatrix2D((focus[0] + 8 * math.sin(t * 0.8), focus[1]), 0, z)
    frame = cv2.warpAffine(frame, C, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if not hasattr(finish, "vign"):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        finish.vign = (1 - 0.45 * np.clip(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2, 0, 1))[..., None]
    frame = frame * finish.vign + rng.normal(0, 0.018, (H, W, 1)).astype(np.float32)
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    for i, line in enumerate(credit):
        cv2.putText(frame, line, (30, H - 24 - 22 * (len(credit) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def writer(out, start, dur):
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(start), "-t", str(dur), "-i", f"{ROOT}/public/song/song.wav",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out], stdin=subprocess.PIPE)


SONG_CREDIT = "Song: 'silence in the air' by JeffSpeed68 (ccmixter.org, CC BY-NC 3.0) + our drums"


class Stage:
    """Dark stage: back wall gradient, colored light beams from above (one pulses per bar),
    soft bokeh, a floor. Precomputed once; per frame only intensities change."""

    COLORS = [(0.95, 0.45, 0.35), (0.35, 0.55, 1.0), (0.40, 0.95, 0.70), (0.95, 0.35, 0.85)]  # BGR

    def __init__(self):
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        self.base = np.broadcast_to((0.07 + 0.09 * yy) * np.array([1.25, 0.9, 1.0], np.float32), (H, W, 3)).copy()
        self.base[int(H * 0.82):] *= 0.55  # floor edge
        hw, hh = W // 4, H // 4
        self.beams = []
        for i, (x, ang) in enumerate([(0.12, 18), (0.38, 6), (0.62, -6), (0.88, -18)]):
            m = np.zeros((hh, hw), np.float32)
            top = np.array([x * hw, -5])
            L = hh * 1.3
            a = math.radians(90 + ang)
            spread = math.radians(9)
            p1 = top + L * np.array([math.cos(a - spread), math.sin(a - spread)])
            p2 = top + L * np.array([math.cos(a + spread), math.sin(a + spread)])
            cv2.fillConvexPoly(m, np.round(np.array([top + [-2, 0], top + [2, 0], p2, p1])).astype(np.int32), 1.0, cv2.LINE_AA)
            fall = np.clip(1.0 - np.linspace(0, 1, hh, dtype=np.float32) * 0.6, 0, 1)[:, None]
            m = cv2.GaussianBlur(m * fall, (0, 0), 6)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            self.beams.append((m, np.array(self.COLORS[i], np.float32)))

    def frame(self, t, wide=False):
        ph = (t % SPB) / SPB
        beat = math.exp(-ph * 5)
        bar = int(t / BAR) % 4
        img = self.base.copy()
        for i, (m, col) in enumerate(self.beams):
            k = 0.10 + (0.30 * beat if i == bar else 0.04 * math.sin(t * 0.9 + i))
            img += m * col * k * 1.3
        return img


def crowd(frame, t, n=13, h=240, seed=4):
    """Out-of-focus crowd at the bottom (heads and shoulders, some hands up), bobbing on the beat."""
    rng = np.random.default_rng(seed)
    ph = (t % SPB) / SPB
    hs = 4  # draw at quarter size, blur, scale up: soft foreground silhouettes
    m = np.zeros((H // hs, W // hs), np.float32)
    for i in range(n):
        x = (i + rng.uniform(-0.3, 0.3)) * W / (n - 1)
        r = rng.uniform(0.8, 1.2) * h * 0.30
        lag = rng.uniform(0, 0.2)
        bob = 14 * math.exp(-(((ph - lag) % 1) * 5))
        y = H - h * rng.uniform(0.40, 0.62) + bob
        cv2.ellipse(m, (int(x / hs), int((y + r * 2.2) / hs)), (int(r * 1.9 / hs), int(r * 1.4 / hs)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.ellipse(m, (int(x / hs), int(y / hs)), (int(r * 0.85 / hs), int(r / hs)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        if rng.uniform() < 0.35:
            up = 40 * math.exp(-(((ph - lag) % 1) * 4))
            side = rng.choice([-1, 1])
            hx = x + side * r * 1.1
            tip = (hx + side * rng.uniform(5, 30), y - r * 1.8 - up)
            cv2.line(m, (int(hx / hs), int((y + r) / hs)), (int(tip[0] / hs), int(tip[1] / hs)), 1.0, max(1, int(r * 0.45 / hs)), cv2.LINE_AA)
            cv2.circle(m, (int(tip[0] / hs), int(tip[1] / hs)), max(1, int(r * 0.32 / hs)), 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 2.2)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    frame *= 1 - 0.96 * m
    frame += m * np.array([0.020, 0.012, 0.018], np.float32)


# shot list: (start, end, shot); a cut on every bar line (a bar = 2 s at 120 bpm)
SHOTS = [(0, 2, "guitarist"), (2, 4, "wide"), (4, 6, "drummer"), (6, 8, "guitarist"), (8, 10, "wide"),
         (10, 14, "singer"), (14, 16, "singer_close"), (16, 18, "guitarist"), (18, 20, "singer"), (20, 22, "wide"),
         (22, 24, "guitarist"), (24, 26, "drummer"), (26, 30, "singer"), (30, 32, "guitarist"), (32, 34, "drummer"),
         (34, 38, "singer_close"), (38, 40, "wide"), (40, 42, "drummer"), (42, 44, "wide")]


class Video:
    def __init__(self):
        self.stage = Stage()
        self.cast = {}

    def get(self, key):
        if key not in self.cast:
            self.cast[key] = {
                "singer": lambda: Singer(),
                "singer_close": lambda: Singer(scale=0.62, top=-10),
                "drummer": lambda: Drummer(),
                "guitarist": lambda: Guitarist(),
                "w_drummer": lambda: Drummer(scale=0.40, top=150),
                "w_singer": lambda: Singer(scale=0.28, cx=W * 0.24, top=470),
                "w_guitarist": lambda: Guitarist(scale=0.42, cx=W * 0.76, top=int(330 - 120 * 0.42)),
            }[key]()
        return self.cast[key]

    def render(self, t0, t1, out, with_audio=False):
        n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        rng = np.random.default_rng(n0)
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(n0, n1):
            t = f / FPS
            a, b, shot = next(sh for sh in SHOTS if sh[0] <= t < sh[1])
            frame = self.stage.frame(t, shot == "wide")
            if shot == "wide":
                who = [self.get("w_drummer"), self.get("w_singer"), self.get("w_guitarist")]
            else:
                who = [self.get(shot)]
            for w in who:
                if isinstance(w, Singer):  # the mic arm follows the singing even while off camera
                    for g in range(getattr(w, "_f", 0), f):
                        w.up.step(w.sing[min(g, len(w.sing) - 1)])
                    w._f = f + 1
                w.draw(frame, t)
            if shot == "wide":
                crowd(frame, t)
            frame = stage_light(frame, t)
            k = (t - a) / (b - a)
            zoom = (1.0, 1.05) if shot != "wide" else (1.04, 1.0)
            credit = [w.j["credit"] for w in who] + [SONG_CREDIT]
            img = finish(frame, t, int(k * 100), 101, rng, credit, zoom=zoom)
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        ff.wait()
        print(out)


def render_video(parts=2):
    """Render the clip in time slices in parallel, then join them with the song."""
    total = SONG["duration"]
    cuts = [0.0, 22.0, total] if parts == 2 else [0.0, total]
    segs = [f"{ROOT}/out/seg{i}.mp4" for i in range(len(cuts) - 1)]
    procs = [subprocess.Popen([sys.executable, __file__, "segment", str(cuts[i]), str(cuts[i + 1]), segs[i]]) for i in range(len(segs))]
    for p in procs:
        p.wait()
    lst = f"{ROOT}/out/segs.txt"
    open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
    out = f"{ROOT}/out/band-video.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", f"{ROOT}/public/song/song.wav",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
    print(out)


def render_one(who, name, start=8.0, dur=15.0, out=None, frames=None):
    n = int(dur * FPS)
    rng = np.random.default_rng(1)
    out = out or f"{ROOT}/out/band-{name}.mp4"
    ff = None if frames else writer(out, start, dur)
    # warm the springs up to the start time
    if hasattr(who, "up"):
        for f in range(int(start * FPS)):
            who.up.step(who.sing[f])
    for f in range(n):
        t = start + f / FPS
        ph = (t % SPB) / SPB
        frame = warp.background("stage", t, math.exp(-ph * 6))
        who.draw(frame, t)
        frame = stage_light(frame, t)
        img = finish(frame, t, f, n, rng, [who.j["credit"], SONG_CREDIT])
        if frames is not None and f in frames:
            cv2.imwrite(frames[f], img)
        if ff:
            ff.stdin.write(img.tobytes())
    if ff:
        ff.stdin.close()
        ff.wait()
        print(out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "singer"
    makers = dict(singer=Singer, drummer=Drummer, guitarist=Guitarist)
    if what in makers:
        render_one(makers[what](), what, *[float(a) for a in sys.argv[2:]])
    elif what == "segment":
        Video().render(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    elif what == "video":
        render_video()
