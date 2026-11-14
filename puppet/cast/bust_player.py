"""A guitar or bass player made from a bust drawing (puppet/build_bust.py -> public/parts/<name>):
the drawing as a head/body puppet, stand-in arms as limbs (2-bone IK to the pick and to the
neck), rigged hands, a guitar on a strap. Strums (guitar) or plucks (bass) from the score.

  BustPlayer(song, "blonde", instrument="guitar", scale=0.5, cx=W / 2, top=0, head=1.0)

All placements are in body-drawing pixels (BODY dict), so a new character only needs its numbers."""
import json
import math

import cv2
import numpy as np

import hand as handrig
import props
import warp
from cast.distor import arms
from groove import Groove
from motion import HeadMotion, Spring, decay_since
from props import blit, place
from warp import over_sub

ROOT = warp.ROOT
W, H = warp.W, warp.H

# where the instrument sits on each body (body-drawing px): the strum point, the angle of the neck,
# the guitar's body length, the strap's end on the far shoulder
BODY = {
    "blonde": dict(strum=(1000, 2120), angle=-22, Lb=980, strap_shoulder=(1400, 1360), strap_back=(520, 1700)),
    "bassboy": dict(strum=(470, 1520), angle=-24, Lb=520, strap_shoulder=(660, 930), strap_back=(200, 1150)),
}
GUITAR = dict(neck=1.55, burst=((0.25, 0.55, 0.85), (0.22, 0.20, 0.10)), n_strings=6)
BASS = dict(neck=2.0, burst=((0.20, 0.20, 0.75), (0.08, 0.06, 0.10)), n_strings=4)


class BustPlayer:
    def __init__(self, song, name, instrument="guitar", scale=0.5, cx=W / 2, top=0, head=1.0, seed=3, groove=1.0):
        """groove: how much the body moves with the music (puppet/groove.py; 0 = still)."""
        self.song, self.name, self.inst = song, name, instrument
        parts = f"{ROOT}/public/parts/{name}"
        j = json.load(open(f"{parts}/parts.json"))
        self.j = j
        pad = j["pad"]
        hd = j["head"]
        sx = lambda v: v + pad  # noqa: E731
        cfg = dict(hd, full=f"public/parts/{name}/body.png", src_h=1000, height=1000 * scale,
                   crop_h_src=(H - top) / scale + 80,
                   head_ells=[(sx(x), y, rx, ry) for x, y, rx, ry in hd["head_ells"]],
                   neck=(sx(hd["neck"][0]), sx(hd["neck"][1]), hd["neck"][2], hd["neck"][3]),
                   face=(sx(hd["face"][0]),) + tuple(hd["face"][1:]), pivot=(sx(hd["pivot"][0]), hd["pivot"][1]),
                   mouth=(sx(hd["mouth"][0]),) + tuple(hd["mouth"][1:]),
                   eyes=[(sx(x), y, rx, ry) for x, y, rx, ry in hd["eyes"]])
        self.p = warp.Puppet(cfg)
        self.s = scale
        b = BODY[name]
        self.b = b
        wl = self.p.head.shape[1]
        self.x0, self.top = int(cx - wl / 2), top
        self.arm_l, self.arm_r = arms(j, parts, fold_shadow=0.15)
        self.arm_r.inner = -self.arm_r.inner  # the fretting elbow hangs down, under the neck
        self.shoulders = {k: np.array(v, float) for k, v in j["shoulders"].items()}
        hs = j["hand"]
        st = handrig.Style(skin=np.array(hs["skin_bgr"]) / 255, ink=(0.075, 0.07, 0.08), ink_w=0.03, shade=0.0, flat=True)
        self.pick_hand = handrig.Hand(st, mirror=False)
        self.fret_hand = handrig.Hand(st, mirror=True)
        self.hsize = hs["length"] * scale
        kind = GUITAR if instrument == "guitar" else BASS
        self.g, self.ga = props.guitar(b["Lb"] * scale, **kind)
        self.g_angle = math.radians(b["angle"])
        # strokes from the score: guitar strums (down/up), bass plucks (alternating fingers)
        if instrument == "guitar":
            ev = [(g["t"], 1 if g["dir"] == "down" else -1, 1.0 if g["accent"] else 0.6) for g in song.data["guitar"]]
            self.chords = [(g["t"], g["chord"]) for g in song.data["guitar"]]
        else:
            ev = [(n["t"], 1 if i % 2 == 0 else -1, n["vel"] / 127) for i, n in enumerate(song.data["bass"])]
            self.chords = [(n["t"], n["pitch"] or 40) for n in song.data["bass"]]
        keys = []
        for t, sg, _ in ev:
            keys += [(t - 0.04, -sg), (t + 0.04, sg)]
        keys.sort()
        self.kt = np.array([k[0] for k in keys])
        self.kv = np.array([k[1] for k in keys], float)
        self.hit_t = np.array([e[0] for e in ev])
        self.fret = Spring(0.0, omega=14.0, zeta=0.8)
        acc = [(t, 0.9) for t, _, a in ev if a > 0.9 and (t % song.bar) < 0.08]
        self.groove = Groove(song, seed=seed, amount=groove) if groove else None
        self.motion = HeadMotion(seed, acc, song.duration, song.spb, turn_amp=0.4 * head, tilt_amp=3.0 * head, nod_px=8 * scale / 0.5 * head)

    def credit(self):
        return self.j["credit"]

    def stroke_pos(self, t):
        i = np.searchsorted(self.kt, t) - 1
        if i < 0:
            return 0.0
        if i + 1 >= len(self.kt):
            return float(self.kv[-1])
        t0, t1, v0, v1 = self.kt[i], self.kt[i + 1], self.kv[i], self.kv[i + 1]
        u = np.clip((t - t0) / max(1e-3, t1 - t0), 0, 1)
        if t1 - t0 > 0.3:
            return float(v0 * (1 - u) ** 2 + v1 * u ** 3)
        return float(v0 + (v1 - v0) * u * u * (3 - 2 * u))

    def fret_target(self, t):
        """Place on the neck (0 = body end, 1 = nut) from the chord / note."""
        ci = max(0, np.searchsorted([c[0] for c in self.chords], t + 0.08) - 1)
        c = self.chords[ci][1] if self.chords else 0
        if isinstance(c, str):
            v = (sum(map(ord, c)) % 5) / 4.0
        else:
            v = ((int(c) - 28) % 12) / 11.0
        return float(self.fret.at(v, t))

    def draw(self, frame, t):
        p, s = self.p, self.s
        turn, tilt, nod, blink, sway = self.motion.at(t)
        sway = sway * 35 * p.s
        fig = p.head_frame(turn, 0.0, "ah", 0.0, sway)  # no blink: the eyes are drawn sockets
        gv = self.groove.at(t) if self.groove else dict(rot=0.0, dx=0.0, dy=0.0, nod=0.0)
        u = s / 0.5  # groove units are px at scale 0.5
        nod = nod + gv["nod"] * u
        thh = math.radians(tilt) * p.w_bend
        hx, hy = p.pivot
        ox, oy = p.X - hx, p.Y - hy
        c, sn = np.cos(thh), np.sin(thh)
        fig = cv2.remap(fig, (hx + c * ox + sn * oy).astype(np.float32), (hy - sn * ox + c * oy - nod * p.w_bend).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pump = decay_since(t, self.hit_t, 0.25)[0]
        bob = 3 * s * math.sin(t * 1.7) + 0.4 * nod + 3 * s * pump + gv["dy"] * u
        top = self.top + bob
        x0 = self.x0 + gv["dx"] * u
        pad = self.j["pad"]
        b = self.b
        # the whole body rocks about the hips; everything attached (guitar, strap, arms) follows
        piv_l = (np.array(b["strum"], float) + [pad, 380]) * s  # hips, in the figure's own pixels
        rot = gv["rot"]
        Ml = cv2.getRotationMatrix2D((float(piv_l[0]), float(piv_l[1])), rot, 1.0)
        fig = cv2.warpAffine(fig, Ml, (fig.shape[1], fig.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        org = np.array([x0, top])
        Tb = lambda q: Ml[:, :2] @ (np.asarray(q, float) - org) + Ml[:, 2] + org  # noqa: E731  screen point -> rocked
        W_ = lambda q: Tb(org + (np.asarray(q, float) + np.array([pad, 0])) * s)  # noqa: E731  body px -> screen
        Wp = lambda q: Tb(org + np.asarray(q, float) * s)  # noqa: E731  padded body px (shoulders) -> screen
        g_angle = self.g_angle - math.radians(rot)
        ring, _ = decay_since(t, self.hit_t, 0.35)
        gimg = props.strings(self.g, self.ga, ring, t)
        strum_at = W_(b["strum"])
        M = place(gimg, self.ga["strum"], strum_at, g_angle, 1.0)
        Lb = (self.ga["neck"][0] - self.ga["bridge"][0]) / 0.67
        gx, gy = self.ga["neck"][0] - Lb, self.ga["neck"][1]
        butt = M @ np.array([gx + 0.01 * Lb, gy, 1])
        horn = M @ np.array([gx + 0.86 * Lb, gy - 0.33 * Lb, 1])
        sw = 34 * s
        props.strap(frame, [butt, butt + np.array([10, -60]) * s, W_(b["strap_back"])], sw)
        warp.over(frame, fig, int(round(x0)), int(round(top)))
        shoulder = W_(b["strap_shoulder"])
        props.strap(frame, [horn, horn * 0.4 + shoulder * 0.6 + np.array([-30, 10]) * s, shoulder], sw)
        # the guitar's shadow on the jacket
        gm = cv2.warpAffine(self.g[..., 3], np.vstack([M[0] / 4 + [0, 0, 4 * s], M[1] / 4 + [0, 0, 6 * s]])[:2],
                            (W // 4, H // 4), flags=cv2.INTER_LINEAR)
        gm = cv2.resize(cv2.GaussianBlur(gm, (0, 0), 3 * s + 1), (W, H))[..., None]
        frame *= 1 - 0.35 * gm * (frame.mean(2, keepdims=True) > 0.25)
        blit(frame, gimg, M)
        props.button(frame, butt, 7 * s)
        props.button(frame, horn, 7 * s)
        # fretting arm (viewer's right): IK from the shoulder to the hand on the neck
        nk0 = M @ np.array([self.ga["neck"][0], self.ga["neck"][1], 1])
        nk1 = M @ np.array([self.ga["nut"][0], self.ga["nut"][1], 1])
        Rf = handrig.orient(math.degrees(g_angle) + 180, 25, 30)
        press = decay_since(t, self.hit_t, 0.12)[0]
        cc = (60 + 10 * press, 90 + 8 * press, 45 + 8 * press)
        fp = handrig.POSES["fret"]
        pose = handrig.Pose({k: cc for k in fp.curls}, fp.thumb, fp.thumb_dir, fp.spread, thumb_hidden=True)
        dips = [q["p1"] for q in self.fret_hand.parts(pose) if q.get("seg") == 1 and q.get("grp") in ("index", "middle", "ring")]
        neck_c = np.mean(dips, 0)
        T = Rf @ np.diag([-1.0, 1, 1])
        off = ((T @ np.zeros(3)) - (T @ neck_c))[:2] * self.hsize
        L = self.arm_r
        S = Wp(self.shoulders["right"])
        reach = 0.92 * (L.L1 + L.L2) * s
        fr = np.linspace(0.05, 0.85, 81)
        dist = np.array([np.hypot(*(nk0 + (nk1 - nk0) * f + off - S)) for f in fr])
        ok = fr[dist < reach]
        lo, hi = (float(ok.min()), float(ok.max())) if len(ok) else (0.1, 0.3)
        frac = lo + (hi - lo) * (0.2 + 0.6 * self.fret_target(t))
        at = nk0 + (nk1 - nk0) * frac
        wrist = at + off
        E, Wr = L.solve(S, wrist, s)
        lay, (x, y) = L.render(S, E, Wr, s, (W, H))
        if lay is not None:
            warp.over(frame, lay, x, y)
        img, xy, fr_ = self.fret_hand.render(pose, Rf, self.hsize, at + (Wr - wrist), anchor=neck_c)
        over_sub(frame, img, xy, fr_)
        # picking arm (viewer's left): the pick (or the fingers, on bass) swings across the strings
        sp = self.stroke_pos(t)
        tip = M @ np.array([self.ga["strum"][0] - 0.06 * self.ga["neck_len"] / 1.55, self.ga["strum"][1] + 0.05 * self.ga["neck_len"] * sp / 1.55, 1])
        Rs = handrig.orient(-25 + 8 * sp + math.degrees(g_angle) + 18, -15, 0)
        pick = np.array([0.20, 0.52, -0.10])
        held = [dict(p0=pick, p1=pick + [0.05, 0.05, 0], r0=0.045, r1=0.02, color=(0.25, 0.25, 0.85), gloss=0.5)] if self.inst == "guitar" else []
        wrist = tip - ((Rs @ pick)[:2] - (Rs @ np.zeros(3))[:2]) * self.hsize
        L = self.arm_l
        S = Wp(self.shoulders["left"])
        E, Wr = L.solve(S, wrist, s)
        lay, (x, y) = L.render(S, E, Wr, s, (W, H))
        if lay is not None:
            warp.over(frame, lay, x, y)
        img, xy, fr_ = self.pick_hand.render(handrig.POSES["pick"], Rs, self.hsize, Wr + (tip - wrist), anchor=pick, held=held)
        over_sub(frame, img, xy, fr_)
