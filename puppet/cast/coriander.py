"""Coriander, the guitarist: David Revoy's concept art (CC BY 4.0), arms cut out as limbs,
rigged hands on the neck and with a pick; strums and chord changes from the score."""
import math

import cv2
import numpy as np

import hand as handrig
import props
import warp
from limb import Limb
from motion import Face, HeadMotion, Spring, decay_since
from props import blit, place
from song import Song
from warp import over_sub

ROOT = warp.ROOT
FPS, W, H = warp.FPS, warp.W, warp.H


CORI_HEAD = dict(
    src_h=2070, head_ells=[(690, 330, 240, 225), (760, 590, 190, 170)], neck=(715, 805, 620, 700),
    face=(765, 545, 85, 105), pivot=(760, 680), mouth=(785, 588, 18, 12), chin=622,
    eyes=[(720, 505, 22, 15), (805, 500, 18, 14)], lid_from=45, lash=(40, 25, 25), lash_w=6,
    hair_from=600, turn_px=8)
CORI_ARM_L = [(575, 800), (618, 860), (604, 960), (566, 1100), (528, 1215), (440, 1222), (390, 1172), (398, 1100), (440, 1010), (498, 900)]
CORI_ARM_R = [(900, 750), (952, 870), (1012, 990), (1064, 1112), (1062, 1185), (1004, 1205), (958, 1150), (928, 1050), (898, 950), (878, 850)]
CORI_ARM_L_J = dict(shoulder=(565, 820), elbow=(500, 990), wrist=(450, 1195))
CORI_ARM_R_J = dict(shoulder=(905, 810), elbow=(975, 990), wrist=(1030, 1180))
CORI_STRUM_ARM = [(375, 1060), (520, 990), (585, 1060), (560, 1180), (535, 1300), (395, 1310), (360, 1190)]
CORI_STRUM_HAND = [(408, 1125), (470, 1118), (503, 1130), (512, 1195), (520, 1240), (505, 1275), (445, 1275), (430, 1200), (405, 1170)]
CORI_FRET_HAND = [(948, 1112), (1045, 1108), (1058, 1160), (1040, 1222), (975, 1222), (948, 1170)]
CHORD_POS = {"D": 0.0, "G": 1.0, "C": 0.55}


def soft_poly(shape, pts, s, sigma):
    m = np.zeros(shape, np.float32)
    cv2.fillPoly(m, [np.round(np.array(pts) * s).astype(np.int32)], 1.0, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), max(0.5, sigma * s)) if sigma > 0 else m


class Guitarist:
    def __init__(self, song=None, scale=0.78, cx=W / 2, top=None, head=1.0, face=False):
        self.song = song or Song.default()
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
        # her painted arms become bending limbs (cut out, the coat behind them inpainted); her painted
        # hands (and the bottle) are painted out and replaced by rigged hands
        img = self.p.head
        arms = {}
        hole = np.zeros(sh, np.uint8)
        for side, poly in (("l", CORI_ARM_L), ("r", CORI_ARM_R)):
            m = soft_poly(sh, poly, s, 1.2)
            a = img.copy()
            J = CORI_ARM_L_J if side == "l" else CORI_ARM_R_J
            d = np.hypot(self.p.X - J["shoulder"][0] * s, self.p.Y - J["shoulder"][1] * s) / s
            if side == "r":  # fretting arm: only the forearm moves (about the painted elbow)
                m = m * np.clip((self.p.Y / s - (J["elbow"][1] - 15)) / 20, 0, 1)
            a[..., 3] *= m * np.clip((d - 40) / 70, 0, 1)  # the shoulder end fades into the coat
            arms[side] = a
            hole |= (m > 0.05).astype(np.uint8)
        for poly in ([(372, 1180), (560, 1180), (560, 1320), (372, 1325)], [(930, 1150), (1085, 1150), (1085, 1255), (930, 1255)]):
            cv2.fillPoly(hole, [np.round(np.array(poly) * s).astype(np.int32)], 1)  # hands, bottle
        hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
        bgr = np.clip(img[..., :3] * 255, 0, 255).astype(np.uint8)
        fill = cv2.inpaint(bgr, hole, 15, cv2.INPAINT_TELEA).astype(np.float32) / 255
        # the coat's own outline where the arms were: the body's alpha without the arms
        body_a = img[..., 3] * (1 - hole)
        body_a = np.maximum(body_a, cv2.erode(img[..., 3], np.ones((int(60 * s) | 1, int(60 * s) | 1), np.uint8)) * hole)
        body_a = cv2.GaussianBlur(body_a, (0, 0), 1.0)
        soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2)[..., None]
        img[..., :3] = img[..., :3] * (1 - soft) + fill * soft
        img[..., 3] = body_a
        self.limbs = {}
        for side, J, inner in (("l", CORI_ARM_L_J, 1), ("r", CORI_ARM_R_J, 1)):
            jj = {k: [v[0] * s, v[1] * s] for k, v in J.items()}
            self.limbs[side] = Limb(arms[side], arms[side], jj, 70 * s, inner=inner, fold_shadow=0.15)
        skin = (106, 159, 196)
        st = handrig.Style(skin=np.array(skin) / 255, ink=(0.14, 0.11, 0.11), ink_w=0.017, shade=0.5)
        self.strum_hand = handrig.Hand(st, mirror=False)
        self.fret_hand = handrig.Hand(st, mirror=True)
        self.hsize = 160 * s
        self.strum_wrist = np.array([458, 1188]) * s
        self.strum_elbow = np.array([540, 1000]) * s
        self.g, self.ga = props.guitar(440 * s)
        self.g_angle = math.radians(-18)
        self.strum_at = np.array([480, 1160]) * s
        # strumming keyframes: each strum crosses the strings, starting on the far side
        keys = []
        for g in self.song.data["guitar"]:
            sg = 1 if g["dir"] == "down" else -1
            keys += [(g["t"] - 0.05, -sg), (g["t"] + 0.05, sg)]
        keys.sort()
        self.kt = np.array([k[0] for k in keys])
        self.kv = np.array([k[1] for k in keys], float)
        self.strum_t = np.array([g["t"] for g in self.song.data["guitar"]])
        self.strum_acc = np.array([1.0 if g["accent"] else 0.6 for g in self.song.data["guitar"]])
        self.chords = [(g["t"], g["chord"]) for g in self.song.data["guitar"]]
        self.fret = Spring(0.0, omega=14.0, zeta=0.8)
        self.prev_turn = 0.0
        acc = [(g["t"], 0.9) for g in self.song.data["guitar"] if g["accent"] and (g["t"] % self.song.bar) < 0.08]
        self.motion = HeadMotion(3, acc, self.song.duration, self.song.spb, turn_amp=0.4 * head, tilt_amp=3.0 * head, nod_px=6 * scale * head)
        gt = self.song.data["guitar"]
        self.face = Face(self.song, accents=[g["t"] for g in gt if g.get("accent") and (g["t"] % self.song.bar) < 0.05],
                         hits=[], singalong=0.45, base_smile=0.2, seed=3) if face else None

    def fret_geometry(self, t, fpos, org):
        """Where the fretting hand goes: back of the hand to the audience, fingers up and curled into
        the fretboard, the neck where the fingertips land. Returns pose, rotation, neck anchor, wrist."""
        s = self.s
        M = place(self.g, self.ga["strum"], org + self.strum_at, self.g_angle, 1.0)
        nk0 = M @ np.array([self.ga["neck"][0], self.ga["neck"][1], 1])
        nk1 = M @ np.array([self.ga["nut"][0], self.ga["nut"][1], 1])
        Rf = handrig.orient(math.degrees(self.g_angle) + 180, 25, 30)
        press = decay_since(t, self.strum_t, 0.12)[0]
        c = (60 + 10 * press, 90 + 8 * press, 45 + 8 * press)
        fp = handrig.POSES["fret"]
        pose = handrig.Pose({k: c for k in fp.curls}, fp.thumb, fp.thumb_dir, fp.spread, thumb_hidden=True)
        dips = [q["p1"] for q in self.fret_hand.parts(pose) if q.get("seg") == 1 and q.get("grp") in ("index", "middle", "ring")]
        neck_c = np.mean(dips, 0)  # the last finger joints sit on the neck's center line (tips on the fretboard)
        held = []
        T = Rf @ np.diag([-1.0, 1, 1])
        off = ((T @ np.zeros(3)) - (T @ neck_c))[:2] * self.hsize
        L = self.limbs["r"]
        E = org + L.E0
        # the chord's place on the neck, where her forearm has its drawn length (no stretching)
        fr = np.linspace(0.05, 0.8, 76)
        dist = np.array([np.hypot(*(nk0 + (nk1 - nk0) * f + off - E)) for f in fr])
        frac = float(fr[np.argmin(np.abs(dist - 1.05 * L.L2))]) - 0.08 * fpos
        at = nk0 + (nk1 - nk0) * frac
        return dict(pose=pose, R=Rf, at=at, neck_c=neck_c, held=held, wrist=at + off - org)

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
        ph = (t % self.song.spb) / self.song.spb
        on_beat = math.cos(math.pi * ph) ** 2
        turn, tilt, nod, blink, sway = self.motion.at(t)
        sway = sway * 35 * p.s
        e = self.face.at(t) if self.face else dict(smile=0.0, brow=0.0, mouth=0.0, squint=0.0)
        fig = p.head_frame(turn, e["mouth"], "ah", max(blink, e["squint"]), sway, smile=e["smile"], brow=e["brow"])
        # strumming forearm: rotate about the elbow; fretting hand: slide along the neck
        th = 0.0  # (her arms are limbs now)
        ci = max(0, np.searchsorted([c[0] for c in self.chords], t + 0.08) - 1)
        fpos = float(self.fret.at(CHORD_POS.get(self.chords[ci][1], 0.0), t))
        d_neck = np.array([math.cos(self.g_angle), math.sin(self.g_angle)])
        X, Y = p.X, p.Y
        px, py = self.piv_strum
        ws = self.w_strum * th
        dx = (X - px) * (np.cos(ws) - 1) - (Y - py) * np.sin(ws)
        dy = (X - px) * np.sin(ws) + (Y - py) * (np.cos(ws) - 1)
        mx, my = (X - dx).astype(np.float32), (Y - dy).astype(np.float32)
        fig = cv2.remap(fig, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # head tilt / nod
        thh = math.radians(tilt) * p.w_bend
        hx, hy = p.pivot
        ox, oy = X - hx, Y - hy
        c, sn = np.cos(thh), np.sin(thh)
        fig = cv2.remap(fig, (hx + c * ox + sn * oy).astype(np.float32), (hy - sn * ox + c * oy - nod * p.w_bend).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bob = 3 * s * math.sin(t * 1.6) + 0.4 * nod
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
        f_ = self.fret_geometry(t, fpos, org)
        pose, Rf, fret_at, neck_c, held = f_["pose"], f_["R"], f_["at"], f_["neck_c"], f_["held"]
        # fretting arm: the painted upper arm stays; the forearm turns about the elbow to the hand's
        # wrist and stretches a little if needed
        L = self.limbs["r"]
        E = org + L.E0
        Wr = org + f_["wrist"]
        L.fore_k = float(np.clip(np.hypot(*(Wr - E)) / L.L2, 0.7, 1.5))
        S = E - (L.E0 - L.S0)
        lay, (x, y) = L.render(S, E, Wr, 1.0, (W, H), part="fore")
        if lay is not None:
            warp.over(frame, lay, x, y)
        img, xy, frac = self.fret_hand.render(pose, Rf, self.hsize, fret_at, anchor=neck_c, held=held)
        over_sub(frame, img, xy, frac)
        # picking arm: the pick swings across the strings with each strum
        sp = self.strum_pos(t)
        Mg = place(self.g, self.ga["strum"], org + self.strum_at, self.g_angle, 1.0)
        tip = Mg @ np.array([self.ga["strum"][0] - 0.06 * self.ga["neck_len"] / 1.55, self.ga["strum"][1] + 0.05 * self.ga["neck_len"] * sp / 1.55, 1])
        Rs = handrig.orient(-25 + 8 * sp, -15, 0)
        pick = np.array([0.20, 0.52, -0.10])
        held = [dict(p0=pick, p1=pick + [0.05, 0.05, 0], r0=0.045, r1=0.02, color=(0.25, 0.25, 0.85), gloss=0.5)]
        wrist = tip - ((Rs @ pick)[:2] - (Rs @ np.zeros(3))[:2]) * self.hsize
        L = self.limbs["l"]
        S = org + L.S0
        E, Wr = L.solve(S, wrist, 1.0)
        lay, (x, y) = L.render(S, E, Wr, 1.0, (W, H))
        if lay is not None:
            warp.over(frame, lay, x, y)
        img, xy, frac = self.strum_hand.render(handrig.POSES["pick"], Rs, self.hsize, Wr + (tip - wrist), anchor=pick, held=held)
        over_sub(frame, img, xy, frac)


