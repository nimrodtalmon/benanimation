"""Distor, the singer: Ben's drawing (public/parts/distor), mic in a rigged hand, his drawn
mouth and closed eyes, timed to the vocal track."""
import json
import math

import cv2
import numpy as np

import hand as handrig
import props
import warp
from limb import Limb, join_parts
from motion import HeadMotion, Spring
from song import Song
from warp import load_rgba, over_sub

ROOT = warp.ROOT
FPS, W, H = warp.FPS, warp.W, warp.H


def body_cfg(j, body_file, scale, crop_h_src):
    """Puppet config for Distor's full-body drawing (public/parts/distor): the head landmarks of
    warp.CHARS['distor'], shifted by the side padding of the body drawing."""
    pad = j["pad"]
    base = warp.CHARS["distor"]
    sx = lambda v: v + pad  # noqa: E731
    cfg = dict(base)
    cfg.update(full="public/parts/distor/" + body_file, src_h=1600, height=1600 * scale, crop_h_src=crop_h_src,
               head_ells=[(sx(x), y, rx, ry) for x, y, rx, ry in base["head_ells"]],
               neck=(sx(base["neck"][0]), sx(base["neck"][1]), base["neck"][2], base["neck"][3]),
               face=(sx(base["face"][0]),) + tuple(base["face"][1:]), pivot=(sx(base["pivot"][0]), base["pivot"][1]),
               mouth=(sx(base["mouth"][0]),) + tuple(base["mouth"][1:]),
               eyes=[(sx(x), y, rx, ry) for x, y, rx, ry in base["eyes"]], lid_color=(sx(base["lid_color"][0]), base["lid_color"][1]),
               irises=[(sx(x), y, r) for x, y, r in base.get("irises", [])],
               eye_hide=[(sx(x), y, rx, ry) for x, y, rx, ry in base.get("eye_hide", [])])
    return cfg


def arms(j, parts, **kw):
    """Distor's two arms (Limbs): his drawn upper arm + forearm, mirrored for the other side."""
    up = load_rgba(f"{parts}/{j['upper_arm']['file']}")
    fo = load_rgba(f"{parts}/{j['forearm']['file']}")
    u, f, jt = join_parts(up, fo, j["upper_arm"], j["forearm"])
    arm_l = Limb(u, f, jt, j["arm_width"], inner=-1, **kw)  # viewer's left arm, bends toward the body
    uf, ff = u[:, ::-1].copy(), f[:, ::-1].copy()  # mirrored copy for the other arm
    wj = u.shape[1] - 1
    jm = {k: [wj - v[0], v[1]] for k, v in jt.items()}
    return arm_l, Limb(uf, ff, jm, j["arm_width"], inner=1, **kw)


class Singer:
    PARTS = f"{ROOT}/public/parts/distor"

    def __init__(self, song=None, scale=0.5, cx=W / 2, top=60, walk=None, stand=None, pump_after=20.0, fluid=False, head=1.0, mouth_twos=False):
        """walk: a motion.Path of his center x on screen (he walks; separate legs). stand: dict(x, grab)
        - a mic stand at screen x; the mic is on it until time `grab`, when he takes it.
        pump_after: from this time the free arm pumps on beats 1 and 3 while singing.
        fluid: weight shift, overlapping arms (follow-through), hand drag, lean and settle.
        head: how much the head moves (1 = default; Clint Eastwood holds the head almost still).
        mouth_twos: mouth shapes change every second frame (drawn-animation lip sync)."""
        self.song = song or Song.default()
        self.path, self.stand_cfg, self.pump_after, self.fluid = walk, stand, pump_after, fluid
        self.mouth_twos = mouth_twos
        from motion import ArmFollow, Drag, Lean
        self.follow = {"l": ArmFollow(bend=1.0), "r": ArmFollow(bend=-1.0)}
        self.drag = {"l": Drag(), "r": Drag()}
        self.lean = Lean()
        j = json.load(open(f"{self.PARTS}/parts.json"))
        cfg = body_cfg(j, "body_nolegs.png" if walk else "body.png", scale, (H - top) / scale + 120)
        self.p = warp.Puppet(cfg)
        self.s = scale
        self.cfg, self.j = cfg, j
        hl, wl = self.p.head.shape[:2]
        self.wl = wl
        self.x0, self.top = int(cx - wl / 2), top
        self.legs = None
        if walk is not None:
            from motion import Walk
            self.legs = {}
            for side in ("l", "r"):
                lj = j["legs"][side]
                img = load_rgba(f"{self.PARTS}/{lj['file']}")
                if fluid:  # round the top of the thigh about the hip joint (no corners poking out when it swings)
                    img = img.copy()
                    hy, hx = lj["hip"][1], lj["hip"][0]
                    yy, xx = np.mgrid[0:hy, 0:img.shape[1]]
                    r = np.hypot(xx - hx, yy - hy)
                    keep = np.clip((0.56 * j["leg_width"] - r) / 12.0 + 0.5, 0, 1)
                    img[:hy, :, 3] = (img[:hy, :, 3] * keep).astype(img.dtype)
                jj = dict(shoulder=lj["hip"], elbow=lj["knee"], wrist=lj["ankle"])
                self.legs[side] = (Limb(img, img, jj, j["leg_width"], inner=1 if side == "l" else -1, fold_shadow=0.0), lj["hip_body"])
            L = self.legs["l"][0]
            leg_px = (L.L1 + L.L2) * scale
            if fluid:  # 3/4 walk: planted feet carry him, knees bend the way he walks
                self.walker = Walk(walk, stride=0.9 * leg_px, lift=0.07 * leg_px, stance=0.6, leg=leg_px, v_full=0.6 * leg_px)
            else:
                self.walker = Walk(walk, stride=0.75 * leg_px, lift=0.16 * leg_px)
            self._dir = 1.0
        self.arm_l, self.arm_r = arms(j, self.PARTS)
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
        acc = self.song.vocal_accents()
        if fluid:  # head accents land a few frames before the stressed syllable
            acc = [(ta - 0.1, a) for ta, a in acc]
        self.settle = Spring(0.0, omega=7.0, zeta=0.4)  # lags walking speed: dip + overshoot at a stop
        self.motion = HeadMotion(1, acc, self.song.duration, self.song.spb, turn_amp=0.5 * head, tilt_amp=3.0 * head, nod_px=6 * scale / 0.5 * head)
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

    def set_body(self, pivot, torso, hips, shift):
        """Body pose for this frame (fluid mode): the torso turns by `torso` and the hips by `hips`
        (radians) about the pelvis `pivot`, then everything shifts by `shift` (screen px)."""
        self.b_piv, self.b_shift = np.asarray(pivot, float), np.asarray(shift, float)
        self.b_R = np.array([[math.cos(torso), -math.sin(torso)], [math.sin(torso), math.cos(torso)]])
        self.b_Rh = np.array([[math.cos(hips), -math.sin(hips)], [math.sin(hips), math.cos(hips)]])

    def X(self, p, hips=False):
        if not self.fluid:
            return np.asarray(p, float)
        R = self.b_Rh if hips else self.b_R
        return self.b_piv + R @ (np.asarray(p, float) - self.b_piv) + self.b_shift

    def draw_legs(self, frame, t, bob):
        """Each foot from the walk cycle; the knee comes toward us when the foot lifts (both bones
        foreshortened, a slight outward knee), the leg drawn behind the jacket."""
        s = self.s
        feet, _, _ = self.walker.feet(t)
        for side in ("l", "r"):
            L, hip_body = self.legs[side]
            hip = self.X(self.world(hip_body, bob), hips=True)
            fx, lift = feet[side]
            if self.fluid:
                off = self.world(hip_body, 0)[0] - self.cx
                foot = np.array([fx + 0.9 * off, self.world(hip_body, 0)[1] + (L.L1 + L.L2) * s - lift * self.walker.lift])
                from motion import knee3q, soft_reach
                foot = soft_reach(hip, foot, (L.L1 + L.L2) * s, start=0.97)
                E, L.up_k, L.fore_k = knee3q(hip, foot, L.L1 * s, L.L2 * s, self._dir)
                self.trace.update({f"hip_{side}": hip, f"knee_{side}": E, f"foot_{side}": foot})
                lay, (x, y) = L.render(hip, E, foot, s, (W, H))
                if lay is not None:
                    warp.over(frame, lay, x, y)
                continue
            # front view: the foot's travel along the walk is mostly toward/away from the camera, so
            # only a quarter of it shows sideways; the lift shows as a raised, shortened leg
            fx = 0.25 * (fx - self.cx) + hip[0]
            leg = (L.L1 + L.L2) * s
            if self.fluid:  # feet stay on the floor; the hips bob above them
                floor = self.world(hip_body, 0)[1] + leg
                foot = np.array([fx, floor - 0.22 * leg * lift])
                if np.hypot(*(foot - hip)) > 0.995 * leg:
                    foot = hip + (foot - hip) * 0.995 * leg / np.hypot(*(foot - hip))
            else:
                foot = np.array([fx, hip[1] + leg * (1 - 0.22 * lift)])
            v = foot - hip
            d = np.hypot(*v)
            out = (-1 if side == "l" else 1) * 0.04 * leg * lift
            E = hip + v * L.L1 / (L.L1 + L.L2) + np.array([out, 0])
            L.up_k = np.hypot(*(E - hip)) / (L.L1 * s)
            L.fore_k = np.hypot(*(foot - E)) / (L.L2 * s)
            lay, (x, y) = L.render(hip, E, foot, s, (W, H))
            if lay is not None:
                warp.over(frame, lay, x, y)

    def stand_mic(self, frame, t, bob):
        """The stand (drawn behind him) and, before the grab, the mic in its clip."""
        st = self.stand_cfg
        s = self.s
        floor = self.world((0, self.j["sole_y"]), 0)[1]
        top = self.world((0, 860), 0)[1]  # clip at mouth height
        x = st["x"]
        props.mic_stand(frame, x, top, floor, s)
        if t < st["grab"]:
            phi = math.radians(-90 + 12)
            yaw = math.degrees(phi - self.mic_a0)
            R = handrig.orient(yaw, 0, 180)
            g = np.array([0.0, 0.44, -0.115])
            img, xy, frac = self.mic_hand.render(handrig.POSES["mic"], R, self.hand_size, (x, top), anchor=g, held=self.mic, show_hand=False)
            over_sub(frame, img, xy, frac)
        return np.array([x, top])

    def draw(self, frame, t):
        self.trace = {}  # key points of this frame (for smoothness checks: tools/jerk.py)
        p, cfg = self.p, self.cfg
        walk_bob, speed = 0.0, 0.0
        if self.path is not None:
            self.cx = self.path.x(t)
            self.x0 = int(self.cx - self.wl / 2)
            _, walk_bob, speed = self.walker.feet(t)
        stand_top = self.stand_mic(frame, t, 0) if self.stand_cfg else None
        gb = dict(sway_px=0.0, bob_px=0.0, hip_tilt=0.0, shoulder_tilt=0.0, speed=0.0)
        lean = 0.0
        if self.fluid and self.path is not None:
            from motion import gait_body
            gb = gait_body(self.walker, t)
            from motion import walk3q
            v = self.path.v(t)
            if abs(v) > 5:
                self._dir = float(np.sign(v))
            hips = {k: self.cx + 0.9 * (self.world(hb, 0)[0] - self.cx) + gb["sway_px"] for k, (_, hb) in self.legs.items()}
            gb["bob_px"] = walk3q(self.walker, t, hips)
            lean = self.lean.at(self.path.v(t) / max(self.s, 1e-3) * 0.245, t)
        f = min(int(t * FPS), len(self.env) - 1)
        fm = min(f + 2, len(self.env) - 1) if self.fluid else f  # mouth shapes lead the audio by 2 frames
        if self.mouth_twos:
            fm -= fm % 2
        if self.fluid and self.path is not None:  # stopping: the body sinks, overshoots, settles
            gb["bob_px"] += 0.04 * self.walker.stride / 0.75 * float(self.settle.at(gb["speed"], t) - gb["speed"])
        ph = (t % self.song.spb) / self.song.spb
        on_beat = math.cos(math.pi * ph) ** 2
        singing = self.sing[f]
        turn, tilt, nod, blink, sway = self.motion.at(t)
        if self.fluid:  # the head stays level while the body tilts
            tilt -= 0.7 * math.degrees(gb["shoulder_tilt"] + lean)
        if self.fluid:  # moving hold: the head never quite freezes
            turn += 0.05 * math.sin(t * 0.83) + 0.03 * math.sin(t * 1.91 + 1.0)
            tilt += 0.6 * math.sin(t * 0.61 + 2.0)
        turn *= 0.7 if singing else 1.0  # sings mostly toward the mic
        if self.path is not None:  # looks where he walks
            turn = float(turn * (1 - speed) + 0.45 * np.sign(self.path.v(t)) * speed)
        sway = sway * 40 * p.s + 3 * p.s * math.sin(t * 2.3)
        fig = p.head_frame(turn, float(self.env[fm]), self.vowels[fm], blink, sway)
        th = math.radians(tilt) * p.w_bend
        px, py = p.pivot
        ox, oy = p.X - px, p.Y - py
        c, sn = np.cos(th), np.sin(th)
        mapx = px + c * ox + sn * oy
        mapy = py - sn * ox + c * oy - nod * p.w_bend
        fig = cv2.remap(fig, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bob = 2.5 * math.sin(t * 1.7) + 0.35 * nod - walk_bob  # breathing, a nod, the walk's bounce
        if self.fluid:
            pad = self.j["pad"]
            self.set_body(self.world((pad + 575, 2350), bob), gb["shoulder_tilt"] + lean, gb["hip_tilt"], (gb["sway_px"], gb["bob_px"]))
        if self.legs:
            self.draw_legs(frame, t, bob)
        if self.fluid:
            R, o = self.b_R, np.array([self.x0, self.top + bob], float)
            M = np.hstack([R, (R @ (o - self.b_piv) + self.b_piv + self.b_shift)[:, None]])
            props.blit(frame, fig, M)
        else:
            warp.over(frame, fig, self.x0, int(self.top + bob))
        # where the mouth is now (after turn, tilt and nod)
        mx, my, mw, J = p.mouth
        q = np.array([mx + turn * cfg["turn_px"] * p.s * 2, my + 0.5 * J * self.env[fm]])
        a = math.radians(tilt)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        mouth = self.X(np.array([self.x0, self.top + bob]) + np.array([px, py]) + R @ (q + [0, nod] - [px, py]))
        self.trace.update(mouth=mouth, shoulder_l=self.X(self.world(self.j["shoulders"]["left"], bob)))
        s = self.s
        shL = self.X(self.world(self.j["shoulders"]["left"], bob))
        shR = self.X(self.world(self.j["shoulders"]["right"], bob))
        # the free arm (viewer's right): hangs and swings; in the chorus it pumps on beats 1 and 3
        chorus = t >= self.pump_after
        beat_i = int(t / self.song.spb) % 4
        pump = chorus and singing and beat_i in (0, 2) and ph < 0.5
        swing = 0.0
        if self.path is not None:  # arms swing against the legs
            swing = 160 * speed * math.sin(2 * math.pi * self.path.x(t) / self.walker.stride)
        free_t = shR + np.array([-60, 980]) * s + np.array([12 * math.sin(t * 2.0) + swing, 0]) * s
        if pump:
            free_t = shR + np.array([-180, 420 - 160 * math.sin(math.pi * ph * 2)]) * s
        if self.fluid:
            if self.path is not None:  # front view: a forward swing lifts the hand toward us and in
                A = (self.arm_r.L1 + self.arm_r.L2) * s
                fr_ = self.walker.feet(t)[0]["r"][0]  # the arm swings forward as its own-side leg goes back
                sw = float(np.clip(-(fr_ - self.cx) / (0.5 * self.walker.stride), -1, 1)) * gb["speed"]
                a = math.radians(4 + 14 * sw * self._dir)
                free_t = shR + 0.96 * A * np.array([-math.sin(a), math.cos(a)])
            E, Wr, a2, om = self.follow["r"].at(shR, free_t, self.arm_r.L1 * s, self.arm_r.L2 * s, t, stiff=0.6 if pump else 0.0)
            drag = self.drag["r"].at(om, t)
            fa = math.degrees(a2 + drag) - 90
            curl = 0.15 + 0.1 * math.sin(t * 1.3) + 0.6 * abs(drag)
            pose = handrig.POSES["fist"] if pump else handrig.Pose.mix(handrig.POSES["relaxed"], handrig.POSES["fist"], min(0.6, curl))
            self.trace.update(elbow_r=E, hand_r=Wr)
            img, xy, frac = self.free_hand.render(pose, handrig.orient(fa, 10, 0), self.hand_size, Wr, anchor=self.anchor)
        else:
            if self.free is None:
                self.free = Spring(free_t, omega=6.0, zeta=0.95)
            wt = self.free.at(free_t, t)
            E, Wr = self.arm_r.solve(shR, wt, s)
            fa = math.degrees(math.atan2(Wr[1] - E[1], Wr[0] - E[0])) - 90
            img, xy, frac = self.free_hand.render(handrig.POSES["fist" if pump else "relaxed"], handrig.orient(fa, 10, 0), self.hand_size, Wr, anchor=self.anchor)
        over_sub(frame, img, xy, frac)
        lay, (x, y) = self.arm_r.render(shR, E, Wr, s, (W, H))
        if lay is not None:
            warp.over(frame, lay, x, y)
        L = self.arm_l
        if self.stand_cfg and t < self.stand_cfg["grab"]:
            # empty hand: swings while walking, reaches for the mic on the stand before the grab
            g0 = self.stand_cfg["grab"]
            reach = float(np.clip((t - (g0 - 0.7)) / 0.7, 0, 1))
            reach = reach * reach * (3 - 2 * reach)
            hang = shL + np.array([60, 980]) * s + np.array([-swing * s, 0])
            tgt = hang * (1 - reach) + (stand_top + np.array([0, 0.25 * self.hand_size])) * reach
            if not hasattr(self, "_reach"):
                self._reach = Spring(tgt, omega=7.0, zeta=0.95)
            wt = self._reach.at(tgt, t)
            E, Wr = L.solve(shL, wt, s)
            fa = math.degrees(math.atan2(Wr[1] - E[1], Wr[0] - E[0])) - 90
            pose = handrig.Pose.mix(handrig.POSES["relaxed"], handrig.POSES["mic"], 0.6 * reach)
            img, xy, frac = self.mic_hand.render(pose, handrig.orient(fa, 10, 0), self.hand_size, Wr, anchor=self.anchor)
            over_sub(frame, img, xy, frac)
            lay, (x, y) = L.render(shL, E, Wr, s, (W, H))
            if lay is not None:
                warp.over(frame, lay, x, y)
            return
        # the mic arm: up to the mouth while singing, down at the chest in pauses
        u = float(np.clip(self.up.step(self.sing[f]), 0, 1))
        u = u * u * (3 - 2 * u)
        head_up = mouth + np.array([-0.10 * mw, 1.45 * mw])
        head_dn = mouth + np.array([-2.0 * mw, 5.0 * mw])
        if self.fluid and self.path is not None:  # the mic hand swings too while he walks
            A = (self.arm_l.L1 + self.arm_l.L2) * s
            fl_ = self.walker.feet(t)[0]["l"][0]
            fwd = float(np.clip(-(fl_ - self.cx) / (0.5 * self.walker.stride), -1, 1)) * gb["speed"] * self._dir
            head_dn = head_dn + np.array([0.10 * A * fwd, -0.06 * A * abs(fwd)])
        head = head_dn * (1 - u) + head_up * u
        if self.stand_cfg:  # just after the grab: the mic comes from the stand
            k = float(np.clip((t - self.stand_cfg["grab"]) / 0.6, 0, 1))
            k = k * k * (3 - 2 * k)
            head = stand_top * (1 - k) + head * k
        phi = math.radians(-90 + (-5 * (1 - u) + 22 * u) + 4 * math.sin(t * 1.3))  # mic axis on screen (up = -90)
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
        if self.fluid:  # overlap: the arm trails the wanted pose (stiff while singing into the mic)
            E, Wr2, a2, om = self.follow["l"].at(shL, Wr, L.L1 * s, L.L2 * s, t, stiff=0.3 + 0.7 * u)
            Wr = Wr2
            R = handrig.orient(yaw + math.degrees(self.drag["l"].at(om, t)) * (1 - u), 0, 180)
        self.trace.update(elbow_l=E, hand_l=Wr)
        held = [dict(q, p0=g + rot @ (np.array(q["p0"]) - g), p1=g + rot @ (np.array(q["p1"]) - g)) for q in self.mic]
        img, xy, frac = self.mic_hand.render(handrig.POSES["mic"], R, self.hand_size, Wr, anchor=self.anchor, held=held)
        over_sub(frame, img, xy, frac)
        lay, (x, y) = L.render(shL, E, Wr, s, (W, H))
        if lay is not None:
            warp.over(frame, lay, x, y)


