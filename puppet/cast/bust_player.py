"""A guitar or bass player made from a bust drawing (public/rigs/<name>.json -> puppet/build_bust.py ->
public/parts/<name>; the instrument's place on the body is the rig's "play" entry):
the drawing as a head/body puppet, stand-in arms as limbs (2-bone IK to the pick and to the
neck), rigged hands, a guitar on a strap. Strums (guitar) or plucks (bass) from the score.

  BustPlayer(song, "blonde", scale=0.5, cx=W / 2, top=0, head=1.0)

Nothing here is specific to a character: all placements come from its parts.json."""
import json
import math

import cv2
import numpy as np

import hand as handrig
import props
import warp
from cast.distor import arms
from groove import Groove
from motion import Face, HeadMotion, Spring, decay_since
from props import blit, place
from warp import over_sub

ROOT = warp.ROOT
W, H = warp.W, warp.H

GUITAR = dict(neck=1.55, burst=((0.25, 0.55, 0.85), (0.22, 0.20, 0.10)), n_strings=6)
BASS = dict(neck=2.0, burst=((0.20, 0.20, 0.75), (0.08, 0.06, 0.10)), n_strings=4)


class BustPlayer:
    def __init__(self, song, name, instrument=None, scale=0.5, cx=W / 2, top=0, head=1.0, seed=3, groove=1.0, style="rock",
                 face=True, shouts=()):
        """face: expressions from the music (smile in loud parts, brows up on accents, a shout at
        each time in shouts, e.g. where the band comes back in)."""
        """groove: how much the body moves with the music (puppet/groove.py; 0 = still)."""
        parts = f"{ROOT}/public/parts/{name}"
        j = json.load(open(f"{parts}/parts.json"))
        self.song, self.name, self.inst = song, name, instrument or j["play"].get("instrument", "guitar")
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
        if "lid_color" in hd:
            cfg["lid_color"] = (sx(hd["lid_color"][0]), hd["lid_color"][1])
        self.p = warp.Puppet(cfg)
        self.s = scale
        b = j["play"]
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
        kind = GUITAR if self.inst == "guitar" else BASS
        self.g, self.ga = props.guitar(b["Lb"] * scale, **kind)
        self.g_angle = math.radians(b["angle"])
        # strokes from the score: guitar strums (down/up), bass plucks (alternating fingers)
        if self.inst == "guitar":
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
        self.groove = Groove(song, seed=seed, amount=groove, style=style) if groove else None
        crashes = [d["t"] for d in song.data.get("drums", []) if d["piece"] == 49]
        self.face = Face(song, accents=crashes, hits=list(self.hit_t), base_smile=0.05, seed=seed) if face else None
        self.shouts = np.sort(np.asarray(shouts, float))
        self.blinks = j.get("blink", True)  # drawn eye sockets don't blink
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
        gv0 = self.groove.at(t) if self.groove else {}
        if "turn" in gv0:  # Clint Eastwood: the head only drifts
            turn, tilt, nod, sway = gv0["turn"], gv0["tilt"], 0.0, 0.0
        mouth, smile, brow, squint = 0.0, 0.0, 0.0, 0.0
        if self.face is not None:
            e = self.face.at(t)
            en = gv0.get("energy", 0.5) if gv0 else 0.5
            shout, _ = decay_since(t, self.shouts, 0.45) if len(self.shouts) else (0.0, 0)
            mouth = min(0.85, e["mouth"] + 0.75 * shout)
            smile = min(1.0, e["smile"] + 0.45 * en - 0.3 * shout)
            brow, squint = min(1.0, e["brow"] + 0.6 * shout), e["squint"] + 0.25 * shout
        fig = p.head_frame(turn, mouth, "ah", min(1.0, blink + squint) if self.blinks else 0.0, sway, smile=smile, brow=brow)
        gv = gv0 if gv0 else dict(rot=0.0, dx=0.0, dy=0.0, nod=0.0)
        u = s / 0.5  # groove units are px at scale 0.5
        nod = nod + gv["nod"] * u
        thh = math.radians(tilt) * p.w_bend
        hx, hy = p.pivot
        ox, oy = p.X - hx, p.Y - hy
        c, sn = np.cos(thh), np.sin(thh)
        fig = cv2.remap(fig, (hx + c * ox + sn * oy).astype(np.float32), (hy - sn * ox + c * oy - nod * p.w_bend).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pump = decay_since(t, self.hit_t, 0.25)[0]
        held = "turn" in gv0
        bob = 0.0 if held else 3 * s * math.sin(t * 1.7) + 0.4 * nod + 3 * s * pump
        bob += gv["dy"] * u
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
        # arms first (poses only): the fretting hand on the neck, the picking hand at the strings.
        # The forearm's open cuff ends a little inside the hand (sink), so the hand comes out of it.
        sink = 0.2 * self.hsize
        nk0 = M @ np.array([self.ga["neck"][0], self.ga["neck"][1], 1])
        nk1 = M @ np.array([self.ga["nut"][0], self.ga["nut"][1], 1])
        Rf = handrig.orient(math.degrees(g_angle) + 180, 25, 30)
        press = decay_since(t, self.hit_t, 0.12)[0]
        cc = (60 + 10 * press, 90 + 8 * press, 45 + 8 * press)
        fp = handrig.POSES["fret"]
        pose = handrig.Pose({k: cc for k in fp.curls}, fp.thumb, fp.thumb_dir, fp.spread, thumb_hidden=True)
        dips = [q["p1"] for q in self.fret_hand.parts(pose) if q.get("seg") == 1 and q.get("grp") in ("index", "middle", "ring")]
        neck_c = np.mean(dips, 0)
        Tf = Rf @ np.diag([-1.0, 1, 1])
        off = ((Tf @ np.zeros(3)) - (Tf @ neck_c))[:2] * self.hsize
        dfin = (Tf @ np.array([0, 1.0, 0]))[:2]
        dfin = dfin / (np.linalg.norm(dfin) + 1e-9)
        Lr = self.arm_r
        Sr = Wp(self.shoulders["right"])
        reach = 0.92 * (Lr.L1 + Lr.L2) * s
        fr = np.linspace(0.05, 0.85, 81)
        dist = np.array([np.hypot(*(nk0 + (nk1 - nk0) * f + off + dfin * sink - Sr)) for f in fr])
        ok = fr[dist < reach]
        lo, hi = (float(ok.min()), float(ok.max())) if len(ok) else (0.1, 0.3)
        frac = lo + (hi - lo) * (0.45 + 0.5 * self.fret_target(t))  # well out on the neck, so the forearm runs under it
        at = nk0 + (nk1 - nk0) * frac
        tgt_r = at + off + dfin * sink
        Er, Wrr = Lr.solve(Sr, tgt_r, s)
        fret_at = at + (Wrr - tgt_r)
        sp = self.stroke_pos(t)
        tip = M @ np.array([self.ga["strum"][0] - 0.06 * self.ga["neck_len"] / 1.55, self.ga["strum"][1] + 0.05 * self.ga["neck_len"] * sp / 1.55, 1])
        Rs = handrig.orient(-25 + 8 * sp + math.degrees(g_angle) + 18, -15, 0)
        pick = np.array([0.20, 0.52, -0.10])
        held = [dict(p0=pick, p1=pick + [0.05, 0.05, 0], r0=0.045, r1=0.02, color=(0.25, 0.25, 0.85), gloss=0.5)] if self.inst == "guitar" else []
        dp = (Rs @ np.array([0, 1.0, 0]))[:2]
        dp = dp / (np.linalg.norm(dp) + 1e-9)
        wrist = tip - ((Rs @ pick)[:2] - (Rs @ np.zeros(3))[:2]) * self.hsize
        Ll = self.arm_l
        Sl = Wp(self.shoulders["left"])
        tgt_l = wrist + dp * 0.5 * sink  # the picking hand shows more of itself
        El, Wrl = Ll.solve(Sl, tgt_l, s)
        pick_at = tip + (Wrl - tgt_l)
        # back to front: back strap, body, upper arms (growing out of the jacket's shoulders), front
        # strap, guitar, then each hand under its forearm
        props.strap(frame, [butt, butt + np.array([10, -60]) * s, W_(b["strap_back"])], sw)
        warp.over(frame, fig, int(round(x0)), int(round(top)))
        for L, S_, E_, Wr_ in ((Ll, Sl, El, Wrl), (Lr, Sr, Er, Wrr)):  # sleeves in front of the jacket sides
            lay, (x, y) = L.render(S_, E_, Wr_, s, (W, H), part="upper")
            if lay is not None:
                warp.over(frame, lay, x, y)
        # the fretting forearm runs behind the neck: drawn before the guitar (only the fingers show on top)
        lay, (x, y) = Lr.render(Sr, Er, Wrr, s, (W, H), part="fore")
        if lay is not None:
            warp.over(frame, lay, x, y)
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
        img, xy, fr_ = self.fret_hand.render(pose, Rf, self.hsize, fret_at, anchor=neck_c)
        over_sub(frame, img, xy, fr_)
        img, xy, fr_ = self.pick_hand.render(handrig.POSES["pick"], Rs, self.hsize, pick_at, anchor=pick, held=held)
        over_sub(frame, img, xy, fr_)
        lay, (x, y) = Ll.render(Sl, El, Wrl, s, (W, H), part="fore")
        if lay is not None:
            warp.over(frame, lay, x, y)
