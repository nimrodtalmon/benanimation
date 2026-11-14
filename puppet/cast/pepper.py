"""Pepper, the drummer: David Revoy's model sheet (CC BY 4.0), arms as limbs, rigged hands
with sticks, a code-drawn kit; every hit from the score."""
import json
import math

import cv2
import numpy as np

import hand as handrig
import props
import warp
from limb import Limb
from motion import Face, HeadMotion, decay_since
from props import blit, place
from song import Song
from warp import load_rgba, over_sub

ROOT = warp.ROOT
FPS, W, H = warp.FPS, warp.W, warp.H


PEPPER_HEAD = dict(
    src_h=1400, head_ells=[(404, 220, 330, 200), (403, 420, 150, 160)], neck=(368, 438, 505, 600),
    face=(403, 420, 100, 125), pivot=(403, 560), mouth=(403, 466, 22, 16), chin=518,
    eyes=[(355, 380, 24, 26), (452, 380, 24, 26)], lid_from=70, lash=(45, 30, 30), lash_w=7,
    hair_from=450, turn_px=10)
DRUM = dict(KICK=36, SNARE=38, HAT=42, OPEN_HAT=46, CRASH=49, TOM_HI=50, TOM_MID=47, TOM_LO=45)


class Drummer:
    PARTS = f"{ROOT}/public/parts/pepper"

    def __init__(self, song=None, scale=0.6, cx=W / 2, top=20, empty=False, head=1.0, face=False):
        """empty=True: only the kit (before she sits down)."""
        self.song = song or Song.default()
        self.empty = empty
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
        # an open-handed kit seen from the audience: hi-hat and snare on the viewer's right (played
        # by the near hand, no crossing), rack toms on the bass drum angled toward her, floor tom
        # and ride on the viewer's left
        self.kit = dict(
            kick=(props.kick_front(330 * s, skull), P(403, 1470)),
            floor=(props.drum(160 * s, 185 * s, seed=4), P(95, 1265)),
            tom_mid=(props.drum(112 * s, 100 * s, tilt=0.24, seed=2), P(300, 1065)),
            tom_hi=(props.drum(104 * s, 95 * s, tilt=0.24, seed=1), P(505, 1065)),
            snare=(props.drum(140 * s, 68 * s, chrome=True, tilt=0.34, seed=5), P(610, 1125)),
            hat=(props.cymbal(128 * s), P(740, 1145)),
            hat_low=(props.cymbal(128 * s), P(740, 1163)),
            crash=(props.cymbal(180 * s), P(770, 880)),
            ride=(props.cymbal(200 * s, tilt=0.26), P(40, 975)),
        )
        self.tilt_deg = dict(tom_mid=9, tom_hi=-9, floor=0, snare=-3, kick=0)
        self.floor_y = P(0, 1760)[1]
        self.ao = None
        hs = j["hand"]
        self.hands = {"l": handrig.StickHand(hs["skin_bgr"], 1.12 * hs["length"] * s, mirror=False), "r": handrig.StickHand(hs["skin_bgr"], 1.12 * hs["length"] * s, mirror=True)}
        self.hand_anchor = (0.0, 0.14, 0.0)  # the wrist sits deep inside the wide cuff, so the hand never pops out of a corner
        # which hand plays what: viewer's left (her right hand) = snare and toms, viewer's right = hats and crash
        piece_at = {DRUM["SNARE"]: "snare", DRUM["TOM_HI"]: "tom_hi", DRUM["TOM_MID"]: "tom_mid", DRUM["TOM_LO"]: "floor",
                    DRUM["HAT"]: "hat", DRUM["OPEN_HAT"]: "hat", DRUM["CRASH"]: "crash"}
        hand_of = {"snare": "l", "tom_hi": "r", "tom_mid": "l", "floor": "l", "hat": "r", "crash": "r"}
        self.hits = {"l": [], "r": []}
        for d in sorted(self.song.data["drums"], key=lambda d: d["t"]):
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
        self.times = {k: np.array([d["t"] for d in self.song.data["drums"] if d["piece"] in v]) for k, v in dict(
            kick=[36], snare=[38], hat=[42, 46], open=[46], crash=[49], tom_hi=[50], tom_mid=[47], floor=[45]).items()}
        self.prev_turn = 0.0
        self.prev_fore = {}
        acc = [(d["t"], 0.5 + 0.5 * d["vel"] / 127) for d in self.song.data["drums"] if d["piece"] in (38, 49) and (d["t"] % self.song.bar) < 0.05 + self.song.bar / 2]
        self.motion = HeadMotion(7, acc, self.song.duration, self.song.spb, turn_amp=0.35 * head, tilt_amp=2.5 * head, nod_px=7 * scale * head)
        dr = self.song.data["drums"]
        self.face = Face(self.song, accents=[d["t"] for d in dr if d["piece"] == 49],
                         hits=[d["t"] for d in dr if d["piece"] == 38 and d["vel"] >= 100], base_smile=0.1, seed=7) if face else None

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
        ang_now = ang + sgn * math.radians(45) * lift  # the wrist flicks the stick up
        return tip, ang, ang_now, lift

    def draw(self, frame, t):
        p, s = self.p, self.s
        f = int(t * FPS)
        ph = (t % self.song.spb) / self.song.spb
        kick, _ = decay_since(t, self.times["kick"], 0.12)
        snare, _ = decay_since(t, self.times["snare"], 0.18)
        bounce = 3 * s / 0.72 * kick
        # head: nods on the snare, glances a little toward the busy hand
        turn, tilt, nod, blink, sway = self.motion.at(t)
        sway = sway * 35 * p.s
        e = self.face.at(t) if self.face else dict(smile=0.0, brow=0.0, mouth=0.0, squint=0.0)
        fig = p.head_frame(turn, e["mouth"], "oh", max(blink, e["squint"]), sway, smile=e["smile"], brow=e["brow"])
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
        self._poses = poses
        # back to front: floor shadows, stands, upper arms, body (shaded behind the kit), cymbals, drums, hands, forearms
        kx, ky = self.kit["kick"][1]
        props.floor_shadow(frame, kx, min(H - 5, ky + 300 * s), 520 * s, 60 * s, 0.55)
        sw = max(4, int(8 * s / 0.72))
        for what in ("crash", "hat"):
            x, y = self.kit[what][1]
            props.stand(frame, (x, y), min(H + 30, self.floor_y), sw)
        sx, sy = self.kit["snare"][1]
        props.stand(frame, (sx, sy + 60 * s), min(H + 30, self.floor_y), sw, tilter=False)
        for side in ("l", "r") if not self.empty else ():
            S, E, Wr = poses[side][:3]
            lay, (x, y) = self.arms[side].render(S, E, Wr, s, (W, H), part="upper")
            if lay is not None:
                warp.over(frame, lay, x, y)
        if not self.empty:
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
        rx_, ry_ = self.kit["ride"][1]
        props.stand(frame, (rx_, ry_), min(H + 30, self.floor_y), sw)
        img, anc = self.kit["ride"][0]
        blit(frame, img, place(img, anc["center"], (rx_, ry_), math.radians(6), 1.0))
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
        fx, fy = self.kit["floor"][1]
        fimg = self.kit["floor"][0][0]
        for dxl in (-0.75, 0.0, 0.8):  # floor tom legs
            bx = fx + dxl * fimg.shape[1] / 2
            props.pole(frame, (bx, fy + 150 * s), (bx + dxl * 40 * s, min(H + 20, self.floor_y)), max(3, int(6 * s / 0.72)))
        for tw, ax in (("tom_mid", -55), ("tom_hi", 55)):  # tom arms out of the bass drum
            tx, ty = self.kit[tw][1]
            props.tom_mount(frame, (kx + ax * s, ky - 300 * s), (tx, ty + 75 * s), max(4, int(9 * s / 0.72)))
        for sgn in (-1, 1):  # spurs on the bass drum
            props.pole(frame, (kx + sgn * 280 * s, ky + 120 * s), (kx + sgn * 380 * s, min(H + 20, ky + 330 * s)), max(3, int(7 * s / 0.72)))
        for what in ("floor", "tom_mid", "tom_hi", "snare", "kick"):  # back to front (the snare sits behind the bass drum)
            img, anc = self.kit[what][0]
            x, y = self.kit[what][1]
            hit, _ = decay_since(t, self.times[what], 0.1)
            sc = 1.0 + 0.02 * hit
            anchor = anc.get("head", anc.get("center"))
            blit(frame, img, place(img, anchor, (x, y + 2 * hit), math.radians(self.tilt_deg[what]), sc))
        for side in ("l", "r") if not self.empty else ():
            S, E, Wr, yaw, s_hand, loose = poses[side]
            img, xy, frac = self.hands[side].render(yaw, s_hand, loose, Wr, self.hand_anchor)
            over_sub(frame, img, xy, frac)
            lay, (x, y) = self.arms[side].render(S, E, Wr, s, (W, H), part="fore")
            if lay is not None:
                warp.over(frame, lay, x, y)


