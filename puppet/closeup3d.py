"""Instrument close-ups in 3D (Blender, Cycles): a player's hands on their 3D instrument, played
from the score, shallow focus. Any rig whose "play" names a 3D model (public/instruments/*.json)
gets them, and any drummer at a 3D kit (public/kits/*.json); the hands take the skin and sleeve
colors from the rig's drawing.

  job = closeup_job(song, rig, kind, t0, t1, light="fire", model=None)
  frames = CloseUp(job)          # renders once (cached under out/closeupcache/), then frames.at(t)

Kinds:
- "fret": the fretting hand from above the neck, a finger a fret on the notes' strings, the thumb
  behind the neck, the hand sliding along the neck between positions.
- "pluck": the plucking hand over the pickup (bass), index and middle alternating rest strokes.
- "pick": the picking hand (guitar) strumming the chords down and up with a pick.
- "sticks": a drummer's two hands and sticks on the snare and the hi-hat (model = the kit's name).
Hands: one mesh per hand, made once in a rest pose (Blender's skin modifier over a skeleton of
wrist, palm, three bones a finger and the thumb, smoothed), then bent per frame by linear blend
skinning from bone frames that follow the joints. Skin detail lives on the rest surface (stored
rest position and per-point joint distance, back/palm side, distance to the tip), so wrinkles on
the knuckles, palm creases, pores, redness at the joints and fingertips stay put as the fingers
move. Subsurface scattering, nails with a pale free edge, the rig's sleeve. Fingers reach their
targets by a small planar IK (the last joint follows the middle one). Light presets: "fire"
(warm from below, cool moon rim), "sunset", "stage". Rendered on threes (10 fps) with a slow
push-in; the 2D shot (cast/closeup_shot.py) puts it over the venue's picture out of focus.
"""
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 10  # on threes

# finger skeleton (meters): MCP knuckle (forward, lateral toward the index side), bone lengths, radii
FINGERS = dict(index=((0.090, 0.024), (0.043, 0.026, 0.020), (0.0098, 0.0080)),
               middle=((0.094, 0.004), (0.047, 0.029, 0.021), (0.0102, 0.0083)),
               ring=((0.088, -0.015), (0.044, 0.027, 0.020), (0.0096, 0.0078)),
               pinky=((0.077, -0.032), (0.034, 0.021, 0.018), (0.0085, 0.0068)))
THUMB = ((0.022, 0.028, -0.012), (0.042, 0.032, 0.027), (0.0125, 0.0092))
NAMES = ["index", "middle", "ring", "pinky"]
SPREAD = dict(index=0.12, middle=0.0, ring=-0.1, pinky=-0.22)  # rest abduction (rad)
LEFT_DRUMS, RIGHT_DRUMS = {38, 40}, {42, 44, 46, 49, 51, 57, 59}


# ---------------------------------------------------------------- the job (2D side)
def tuning(pitches, n):
    """Open-string pitches, low to high: standard tuning, lowered so the lowest note is playable."""
    base = [28, 33, 38, 43] if n == 4 else [40, 45, 50, 55, 59, 64]
    low = min(pitches) if len(pitches) else base[0]
    down = int(np.clip(base[0] - low, 0, 7))
    return [p - down for p in base]


def finger_notes(notes, n, frets=12):
    """(t, string, fret) for each note: the string that keeps the hand near where it is."""
    tun = tuning([p for _, p in notes], n)
    pos, out = 3.0, []
    for t, p in notes:
        cand = [(s, p - o) for s, o in enumerate(tun) if 0 <= p - o <= frets]
        if not cand:
            p2 = p
            while p2 < tun[0]:
                p2 += 12
            while p2 - tun[-1] > frets:
                p2 -= 12
            cand = [(s, p2 - o) for s, o in enumerate(tun) if 0 <= p2 - o <= frets] or [(0, 0)]
        s, f = min(cand, key=lambda c: abs((c[1] or pos) - pos) + 0.3 * c[0])
        if f > 0:
            pos = 0.7 * pos + 0.3 * f if abs(f - pos) < 3 else max(1.0, f - 0.5)
        out.append((t, s, f))
    return out


def rig_colors(rig):
    """Skin (from the face) and sleeve (from the upper arm) colors of a rig's drawing, linear RGB."""
    d = f"{ROOT}/public/parts/{rig}"
    j = json.load(open(f"{d}/parts.json"))
    body = cv2.imread(f"{d}/body.png", cv2.IMREAD_UNCHANGED)
    if "head" in j:
        fx, fy, frx, fry = j["head"]["face"][:4]
    else:
        import warp
        fx, fy, frx, fry = warp.CHARS[rig]["face"][:4]
    fx += j.get("pad", 0)
    patch = body[int(fy + 0.15 * fry):int(fy + 0.45 * fry), int(fx - 0.5 * frx):int(fx + 0.5 * frx)]
    px = patch[patch[..., 3] > 200][:, :3].astype(np.float32)
    hsv = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_BGR2HSV)[0]
    keep = (hsv[:, 2] > 90) & (hsv[:, 1] > 25)
    skin = np.median(px[keep] if keep.sum() > 20 else px, 0)[::-1] / 255
    arm = cv2.imread(f"{d}/upper_arm.png", cv2.IMREAD_UNCHANGED)
    sleeve = np.median(arm[arm[..., 3] > 200][:, :3].astype(np.float32), 0)[::-1] / 255
    lin = lambda c: [float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)]  # noqa: E731
    return lin(skin), lin(sleeve)


def closeup_job(song, rig, kind, t0, t1, light="fire", res=(1280, 720), samples=20, model=None):
    j = json.load(open(f"{ROOT}/public/parts/{rig}/parts.json"))
    skin, sleeve = rig_colors(rig)
    base = dict(rig=rig, kind=kind, t0=round(t0, 3), t1=round(t1, 3), fps=FPS, light=light, res=list(res), samples=samples,
                skin=skin, sleeve=sleeve, seed=int(sum(map(ord, rig)) + 7 * int(t0)))
    if kind == "sticks":
        hits = [[round(e["t"], 3), int(e["piece"])] for e in song.data.get("drums", []) if t0 - 1.0 <= e["t"] < t1 + 0.5]
        return dict(base, model=model or "classic", family="kit", notes=hits)
    play = j.get("play", {})
    model = model or play.get("model")
    if model is None:
        return None
    inst = json.load(open(f"{ROOT}/public/instruments/{model}.json"))
    key = "bass" if play.get("instrument") == "bass" or inst["strings"] == 4 else "guitar"
    ev = [e for e in song.data.get(key, []) if t0 - 2.0 <= e["t"] < t1 + 0.5]
    notes = [(e["t"], e.get("pitch", e.get("root", 40))) for e in ev]
    out = dict(base, model=model, family="string", notes=[[round(t, 3), s, f] for t, s, f in finger_notes(notes, inst["strings"])])
    if kind == "pick":
        out["strums"] = [[round(e["t"], 3), e.get("dir", "down"), bool(e.get("accent"))] for e in ev]
    return out


def job_dir(job):
    spec = f"{ROOT}/public/kits/{job['model']}.json" if job.get("family") == "kit" else f"{ROOT}/public/instruments/{job['model']}.json"
    builder = f"{ROOT}/puppet/kit3d.py" if job.get("family") == "kit" else f"{ROOT}/puppet/guitar3d.py"
    src = b"".join(open(f, "rb").read() for f in (os.path.abspath(__file__), builder, spec))
    h = hashlib.sha1(json.dumps(job, sort_keys=True).encode() + src).hexdigest()[:12]
    return f"{ROOT}/out/closeupcache/{job['rig']}-{job['kind']}-{h}"


class CloseUp:
    """The rendered frames of one close-up shot (rendered on first use; safe across processes)."""

    def __init__(self, job):
        self.job, self.d = job, job_dir(job)
        self.frames = None

    def ensure(self):
        if self.frames is not None:
            return
        os.makedirs(self.d, exist_ok=True)
        with open(f"{self.d}/.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not os.path.exists(f"{self.d}/meta.json"):
                json.dump(self.job, open(f"{self.d}/job.json", "w"))
                subprocess.run([sys.executable, os.path.abspath(__file__), "build", f"{self.d}/job.json"], check=True)
        self.meta = json.load(open(f"{self.d}/meta.json"))
        self.frames = {}

    def at(self, t):
        """BGRA float image of the shot at time t."""
        self.ensure()
        i = int(np.clip(round((t - self.job["t0"]) * FPS), 0, self.meta["n"] - 1))
        if i not in self.frames:
            if len(self.frames) > 6:  # keep memory small: each frame is a full float image
                self.frames.clear()
            im = cv2.imread(f"{self.d}/f_{i:04d}.png", cv2.IMREAD_UNCHANGED)
            self.frames[i] = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
        return self.frames[i]


# ---------------------------------------------------------------- hand geometry (numpy)
def _v(*a):
    return np.array(a, float)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def finger_ik(B, fwd, down, lens, T):
    """Planar IK: MCP and PIP flex (DIP = 0.7 PIP) so the tip reaches T. fwd/down span the plane.
    Returns the joint points [B, P1, P2, tip]."""
    rel = T - B
    a, b = float(rel @ fwd), float(rel @ down)
    th1 = np.linspace(-0.05, 1.5, 90)[:, None]  # no bending backward
    th2 = np.linspace(0.15, 1.6, 80)[None, :]
    L1, L2, L3 = lens
    x = L1 * np.cos(th1) + L2 * np.cos(th1 + th2) + L3 * np.cos(th1 + 1.7 * th2)
    y = L1 * np.sin(th1) + L2 * np.sin(th1 + th2) + L3 * np.sin(th1 + 1.7 * th2)
    err = (x - a) ** 2 + (y - b) ** 2 + 2e-6 * (th2 - 0.9) ** 2
    i, j = np.unravel_index(np.argmin(err), err.shape)
    t1, t2 = float(th1[i, 0]), float(th2[0, j])
    d = lambda th: fwd * math.cos(th) + down * math.sin(th)  # noqa: E731
    P1 = B + L1 * d(t1)
    P2 = P1 + L2 * d(t1 + t2)
    P3 = P2 + L3 * d(t1 + 1.7 * t2)
    return [B, P1, P2, P3]


class HandPose:
    """A hand placed by a frame (wrist, forward, lateral toward the index side, dorsal). A right
    hand has (forward x lateral) . dorsal > 0, a left hand < 0. chain()/thumb() give the joint
    points and the normal of the plane the finger bends in."""

    def __init__(self, wrist, fwd, lat, dorsal):
        self.W = _v(*wrist)
        self.f, self.l, self.dz = _unit(_v(*fwd)), _unit(_v(*lat)), _unit(_v(*dorsal))

    def P(self, x, y, z=0.0):
        return self.W + self.f * x + self.l * y + self.dz * z

    geom = None  # a hand's own knuckles and bone lengths (Hand3D sets it), else FINGERS/THUMB

    def chain(self, name, target=None, curl=0.6, spread=0.0):
        if self.geom is not None:
            (bx, by, bz), lens = self.geom[name]
            B = self.P(bx, by, bz)
        else:
            (bx, by), lens, _ = FINGERS[name]
            B = self.P(bx, by)
        if target is None:  # relaxed: curled by `curl`, spread by `spread`
            fw = _unit(self.f * math.cos(spread) + self.l * math.sin(spread))
            down = -self.dz
            ang = [curl * 0.6, curl * 1.0, curl * 0.7]
            pts, th, p = [B], 0.0, B
            for L, a in zip(lens, ang):
                th += a
                p = p + L * (fw * math.cos(th) + down * math.sin(th))
                pts.append(p)
            return pts, np.cross(fw, down)
        T = _v(*target)
        side = (T - B) - ((T - B) @ self.dz) * self.dz  # abduct toward the target, within reason
        fw = _unit(side) if np.linalg.norm(side) > 1e-6 else self.f
        if fw @ self.f < 0.75:
            fw = _unit(self.f * 0.75 + _unit(fw - (fw @ self.f) * self.f) * 0.66)
        down = _unit(-self.dz - ((-self.dz) @ fw) * fw)
        return finger_ik(B, fw, down, lens, T), np.cross(fw, down)

    def thumb(self, target=None):
        (bx, by, bz), lens = self.geom["thumb"] if self.geom is not None else THUMB[:2]
        B = self.P(bx, by, bz)
        T = self.P(0.10, 0.045, -0.04) if target is None else _v(*target)
        fw = _unit(T - B)
        side = _unit(np.cross(fw, self.dz) if abs(fw @ self.dz) < 0.95 else self.l)
        down = _unit(np.cross(side, fw))
        if down @ (-self.dz) < 0:
            down = -down
        return finger_ik(B, fw, down, lens, T), np.cross(fw, down)


def bone_frames(hp, chains):
    """Palm + three per finger + three for the thumb: the frames (4x4) that bend the skin. Each
    frame's y axis keeps the hand's handedness, so rest and pose frames match for left hands too."""
    F = []
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = hp.f, hp.l, hp.dz, hp.W
    F.append(M)
    hand_sign = np.sign(np.dot(np.cross(hp.f, hp.l), hp.dz))
    for pts, n in chains:
        for j in range(3):
            x = _unit(pts[j + 1] - pts[j])
            z = _unit(np.cross(x, n))
            y = np.cross(z, x) * hand_sign
            M = np.eye(4)
            M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, pts[j]
            F.append(M)
    return F


def _quat(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _qmul(a, b):
    """Quaternion products, broadcasting over leading axes."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def dq_skin(Wt, A, V):
    """Dual-quaternion skinning: points V (n, 3), weights Wt (n, bones), rigid transforms A (bones, 4, 4).
    Unlike blending matrices, joints keep their volume when they bend a lot."""
    qr = np.stack([_quat(a[:3, :3]) for a in A])  # (bones, 4)
    tq = np.c_[np.zeros(len(A)), A[:, :3, 3]]
    qd = 0.5 * _qmul(tq, qr)
    ref = qr[np.argmax(Wt, 1)]  # flip quaternions into the dominant bone's hemisphere
    sgn = np.sign(np.einsum("bk,nk->nb", qr, ref))
    sgn[sgn == 0] = 1
    w = Wt * sgn
    br, bd = w @ qr, w @ qd
    nr = np.linalg.norm(br, axis=1, keepdims=True)
    br, bd = br / nr, bd / nr
    vq = np.c_[np.zeros(len(V)), V]
    conj = br * np.array([1, -1, -1, -1])
    rot = _qmul(_qmul(br, vq), conj)[:, 1:]
    t = 2 * _qmul(bd, conj)[:, 1:]
    return rot + t


def _seg_dist(P, a, b):
    ab = b - a
    u = np.clip(((P - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
    q = a + u[:, None] * ab
    return np.linalg.norm(P - q, axis=1), u, q


# ---------------------------------------------------------------- Blender side
class Hand3D:
    """A hand bent per frame (see the module docstring). The mesh, bones and weights come from the
    MakeHuman base mesh (CC0; public/hands/mh_<side>.npz, made by tools/mh_hands.py): real hand
    anatomy (knuckles, finger pads, palm, tendons' shape) instead of tubes."""

    def __init__(self, name, side, mats):
        import bpy
        self.bpy = bpy
        self.side = side
        D = np.load(f"{ROOT}/public/hands/mh_{side}.npz")
        V = D["verts"].astype(float)
        self.Wt = D["weights"].astype(float)
        # the file is in the hand's own frame (forward, index side, back). As world points at rest, a left
        # hand's frame is mirrored (index side -y), so every bone transform stays a rotation.
        lat = (0, 1, 0) if side == "right" else (0, -1, 0)
        rest = HandPose((0, 0, 0), (1, 0, 0), lat, (0, 0, 1))
        V = rest.P(V[:, 0:1], V[:, 1:2], V[:, 2:3]) if False else V * np.array([1.0, lat[1], 1.0])
        self.geom = {}
        rest_chains = []
        for nm in NAMES + ["thumb"]:
            cl = D[f"chain_{nm}"].astype(float)
            lens = tuple(float(np.linalg.norm(cl[j + 1] - cl[j])) for j in range(3))
            self.geom[nm] = (tuple(cl[0]), lens)
            c = cl * np.array([1.0, lat[1], 1.0])
            fw = _unit(c[3] - c[0])
            if nm == "thumb":
                sd = _unit(np.cross(fw, rest.dz))
                down = _unit(np.cross(sd, fw))
                if down @ (-rest.dz) < 0:
                    down = -down
            else:
                fw = _unit(fw - (fw @ rest.dz) * rest.dz)
                down = _unit(-rest.dz - ((-rest.dz) @ fw) * fw)
            rest_chains.append((list(c), np.cross(fw, down)))
        rest.geom = self.geom
        self.handed = 1.0 if side == "right" else -1.0
        faces = []
        uvs = []
        for q, u in zip(D["faces"], D["uv"]):
            q, u = list(q), list(u)
            if q[2] == q[3]:
                q, u = q[:3], u[:3]
            faces.append(q)
            uvs.append(u)
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in V], [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        k = 0
        for poly, u in zip(me.polygons, uvs):
            for li, uv in zip(poly.loop_indices, u):
                uvl.data[li].uv = tuple(uv)
            k += 1
        self.obj = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(self.obj)
        me.materials.append(mats["skin"])
        for p in me.polygons:
            p.use_smooth = True
        sub = self.obj.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = 2
        self.Vh = np.c_[V, np.ones(len(V))]
        self.R_inv = [np.linalg.inv(M) for M in self._frames(rest, rest_chains)]
        self.attributes(me, V, rest, rest_chains)
        self.tip_r = {}
        for nm, (c, _) in zip(NAMES + ["thumb"], rest_chains):  # fingertip thickness, for the nails
            d, u, q = _seg_dist(V, c[2], c[3])
            near = (u > 0.4) & (d < 0.015)
            self.tip_r[nm] = float(np.median(d[near])) if near.any() else 0.007
        self.nails = []
        for k in range(5):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0)
            nl = bpy.context.object
            nl.data.materials.append(mats["nail"])
            bpy.ops.object.shade_smooth()
            self.nails.append(nl)
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.0, depth=1.0)
        self.sleeve = bpy.context.object
        dm = self.sleeve.modifiers.new("d", "DISPLACE")
        tx = bpy.data.textures.new(name + "folds", "CLOUDS")
        tx.noise_scale = 0.35
        dm.texture, dm.strength = tx, 0.12
        self.sleeve.modifiers.new("s", "SUBSURF").render_levels = 2
        self.sleeve.data.materials.append(mats["sleeve"])
        bpy.ops.object.shade_smooth()

    def _frames(self, hp, chains):
        F = bone_frames(hp, chains)
        if self.side == "left":  # the left mesh is stored with its own (mirrored) lateral: keep frames right-handed
            pass
        return F

    def chains(self, hp, targets, thumb_target, curls=None, spread=False):
        hp.geom = self.geom
        out = []
        for k in NAMES:
            out.append(hp.chain(k, targets.get(k), (curls or {}).get(k, 0.7), SPREAD[k] if spread else 0.0))
        out.append(hp.thumb(thumb_target))
        return out

    def attributes(self, me, V, hp, chains):
        """Rest-surface data for the skin shader."""
        n = len(V)
        jd, dors, tip = np.ones(n), np.zeros(n), np.ones(n)
        best = np.full(n, 1e9)
        for c, nrm in chains:
            L = [np.linalg.norm(c[j + 1] - c[j]) for j in range(3)]
            arc = [0.0, L[0], L[0] + L[1], L[0] + L[1] + L[2]]
            for j in range(3):
                d, u, q = _seg_dist(V, c[j], c[j + 1])
                m = d < best
                best[m] = d[m]
                s = arc[j] + u * L[j]
                jd[m] = np.min(np.abs(s[m, None] - np.array(arc[:3])[None, :]), 1)
                x = _unit(c[j + 1] - c[j])
                z = _unit(np.cross(x, nrm))
                r = V[m] - q[m]
                dors[m] = (r @ z) / (np.linalg.norm(r, axis=1) + 1e-9)
                tip[m] = np.linalg.norm(V[m] - c[3], axis=1)
        palm = best > 0.022
        jd[palm], tip[palm] = 1.0, 1.0
        dors[palm] = np.clip((V[palm] - hp.W) @ hp.dz / 0.012, -1, 1)
        for nm, arr in (("jd", jd), ("dors", dors), ("tip", tip)):
            a = me.attributes.new(nm, "FLOAT", "POINT")
            a.data.foreach_set("value", arr.astype(np.float32))
        a = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", V.reshape(-1).astype(np.float32))

    def pose(self, hp, targets, thumb_target=None, curls=None, reach=True):
        """reach: move the hand (not the targets) until the fingertips get to their targets."""
        from mathutils import Matrix, Vector
        chains = self.chains(hp, targets, thumb_target, curls)
        for _ in range(4 if reach and targets else 0):
            miss = [np.asarray(targets[k], float) - chains[i][0][3] for i, k in enumerate(NAMES) if k in targets]
            m = np.mean(miss, 0)
            if np.linalg.norm(m) < 0.002:
                break
            hp.W = hp.W + 0.85 * m
            chains = self.chains(hp, targets, thumb_target, curls)
        A = np.stack([M @ Ri for M, Ri in zip(self._frames(hp, chains), self.R_inv)])  # (bones, 4, 4)
        P = dq_skin(self.Wt, A, self.Vh[:, :3])
        me = self.obj.data
        me.vertices.foreach_set("co", P.reshape(-1).astype(np.float32))
        me.update()
        for nl, (c, nrm), r in zip(self.nails, chains, [self.tip_r[k] for k in NAMES + ["thumb"]]):
            d = _unit(c[3] - c[2])
            up = _unit(np.cross(d, nrm))
            side = np.cross(up, d)
            ctr = c[3] - d * r * 0.75 + up * r * 0.62
            R = Matrix(((d[0], side[0], up[0]), (d[1], side[1], up[1]), (d[2], side[2], up[2])))
            nl.matrix_world = Matrix.Translation(Vector(ctr)) @ R.to_4x4() @ Matrix.Diagonal((r * 0.85, r * 0.62, r * 0.12, 1))
        a, bb = hp.W - hp.f * 0.03, hp.W - hp.f * 0.42
        q = Vector((0, 0, 1)).rotation_difference(Vector(_unit(a - bb)))
        self.sleeve.matrix_world = Matrix.Translation(Vector((a + bb) / 2)) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((0.05, 0.047, 0.39, 1))
        return chains


def hand_materials(job):
    import blendkit as bk
    skin = np.array(job["skin"]) * 0.6
    m, n, b = bk.material("skin", tuple(skin), 0.45)
    b.inputs["Subsurface Weight"].default_value = 0.6
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.25, 0.12)
    b.inputs["Subsurface Scale"].default_value = 0.005
    b.inputs["Coat Weight"].default_value = 0.08
    b.inputs["Coat Roughness"].default_value = 0.35
    attr = lambda nm, out="Fac": n.node("ShaderNodeAttribute", props=dict(attribute_name=nm)).outputs[out]  # noqa: E731
    rest = attr("rest", "Vector")
    jd, dors, tip = attr("jd"), attr("dors"), attr("tip")
    wob = n.math("MULTIPLY", n.math("SUBTRACT", n.node("ShaderNodeTexNoise", Vector=rest, Scale=700.0).outputs[0], 0.5), 0.0006)
    jdw = n.math("ADD", jd, wob)
    near = lambda w: n.math("EXPONENT", n.math("MULTIPLY", n.math("POWER", n.math("DIVIDE", jdw, w), 2.0), -1.0))  # noqa: E731
    back = n.math("MAXIMUM", dors, 0.0)
    front = n.math("MAXIMUM", n.math("MULTIPLY", dors, -1.0), 0.0)
    rings = n.math("ADD", n.math("MULTIPLY", n.math("SINE", n.math("MULTIPLY", jdw, 2 * math.pi / 0.0011)), 0.5), 0.5)
    wrinkles = n.math("MULTIPLY", n.math("MULTIPLY", rings, near(0.0038)), back)
    crease = n.math("MULTIPLY", near(0.0007), front)
    pores = n.node("ShaderNodeTexNoise", Vector=rest, Scale=2600.0, Detail=3.0).outputs[0]
    fine = n.node("ShaderNodeTexVoronoi", Vector=rest, Scale=900.0, props=dict(feature="DISTANCE_TO_EDGE")).outputs[0]
    h = n.math("ADD", n.math("ADD", n.math("MULTIPLY", wrinkles, -0.8), n.math("MULTIPLY", crease, -1.2)),
               n.math("ADD", n.math("MULTIPLY", pores, 0.25), n.math("MULTIPLY", fine, 0.15)))
    n.set(b.inputs["Normal"], n.node("ShaderNodeBump", Strength=0.6, Distance=0.0004, Height=h).outputs[0])
    red = n.math("MINIMUM", n.math("ADD", n.math("MULTIPLY", near(0.006), n.math("ADD", back, 0.3)),
                                   n.math("EXPONENT", n.math("MULTIPLY", n.math("POWER", n.math("DIVIDE", tip, 0.014), 2.0), -1.0))), 1.0)
    mott = n.node("ShaderNodeTexNoise", Vector=rest, Scale=90.0, Detail=4.0).outputs[0]
    mix = n.node("ShaderNodeMix", Factor=n.math("MULTIPLY", red, 0.7), props=dict(data_type="RGBA"))
    n.set(mix.inputs[6], n.ramp(mott, [(0.35, tuple(skin * 0.93)), (0.65, tuple(skin * 1.06))]))
    mix.inputs[7].default_value = (*(skin * np.array([1.12, 0.72, 0.68])), 1)
    n.set(b.inputs["Base Color"], mix.outputs[2])
    n.set(b.inputs["Roughness"], n.math("ADD", 0.38, n.math("MULTIPLY", mott, 0.18)))
    nail, nn, nb = bk.material("nail", tuple(np.clip(skin * np.array([1.05, 0.8, 0.8]), 0, 1)), 0.18)
    nb.inputs["Coat Weight"].default_value = 1.0
    nb.inputs["Coat Roughness"].default_value = 0.08
    gx = nn.xyz(nn.node("ShaderNodeTexCoord").outputs["Generated"])[0]
    nn.set(nb.inputs["Base Color"], nn.ramp(gx, [(0.3, tuple(skin * np.array([1.0, 0.72, 0.72]))), (0.7, tuple(skin * np.array([1.05, 0.85, 0.82]))),
                                                 (0.86, tuple(np.clip(skin * 1.25 + 0.05, 0, 1)))]))
    cloth, n2, b2 = bk.material("sleeve", tuple(job["sleeve"]), 0.85)
    b2.inputs["Sheen Weight"].default_value = 0.6
    weave = n2.node("ShaderNodeTexWave", Vector=n2.node("ShaderNodeTexCoord").outputs["Object"], Scale=180.0)
    n2.set(b2.inputs["Normal"], n2.node("ShaderNodeBump", Strength=0.15, Height=weave.outputs[1]).outputs[0])
    return dict(skin=m, nail=nail, sleeve=cloth)


def lights(preset, tgt):
    import blendkit as bk
    if preset == "fire":
        bk.light("AREA", (tgt.x - 0.1, tgt.y - 0.6, tgt.z - 0.25), 60, (1.0, 0.45, 0.13), size=0.5, target=tgt)
        bk.light("AREA", (tgt.x + 0.2, tgt.y + 0.5, tgt.z + 0.6), 25, (0.55, 0.65, 1.0), size=0.6, target=tgt)
        bk.world(0.03, (0.05, 0.06, 0.12))
    elif preset == "sunset":
        bk.light("AREA", (tgt.x - 0.9, tgt.y - 0.3, tgt.z + 0.15), 70, (1.0, 0.6, 0.3), size=0.6, target=tgt)
        bk.light("AREA", (tgt.x + 0.3, tgt.y + 0.4, tgt.z + 0.8), 30, (0.5, 0.62, 1.0), size=1.0, target=tgt)
        bk.world(0.15, (0.35, 0.4, 0.6))
    else:  # stage
        bk.light("AREA", (tgt.x - 0.5, tgt.y + 0.5, tgt.z + 0.5), 70, (1.0, 0.3, 0.6), size=0.4, target=tgt)
        bk.light("AREA", (tgt.x + 0.5, tgt.y - 0.5, tgt.z + 0.4), 70, (0.3, 0.5, 1.0), size=0.4, target=tgt)
        bk.light("AREA", (tgt.x, tgt.y - 0.4, tgt.z + 0.7), 40, (1, 1, 1), size=0.6, target=tgt)
        bk.world(0.02)


def eased(fn, t, w=0.12):
    return float(np.mean([fn(t + d) for d in np.linspace(-w, 0.04, 7)]))


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- strings, pick, sticks
class Strings:
    """Strings as real tubes (wound ones with windings) from the bridge to the nut; they shake."""

    def __init__(self, sp, g):
        import bpy  # noqa: I001
        import bmesh
        import blendkit as bk
        M = 0.01
        n, top, nut, bx = g["n"], g["top"], g["nut"], g["bx"]
        self.n, self.nut, self.bx, self.frets, self.M = n, nut, bx, g["frets"], M
        self.z_nut = (top + 1.25) * M
        self.z_br = (top + (1.25 if n == 4 else 1.95)) * M
        self.sp_br = 1.5 if n == 4 else 1.05
        self.sp_nut = (g["w_nut"] - 0.8) / (n - 1)
        gauges = [0.105, 0.085, 0.065, 0.045] if n == 4 else [0.046, 0.036, 0.026, 0.017, 0.013, 0.010]
        self.S = []
        for i in range(n):
            wound = gauges[i] > 0.02
            m, nd, b = bk.material(f"str{i}", (0.72, 0.7, 0.66), 0.25 if wound else 0.15, 1.0)
            if wound:
                tc = nd.node("ShaderNodeTexCoord").outputs["Object"]
                wv = nd.node("ShaderNodeTexWave", Vector=tc, Scale=900.0 / max(gauges[i], 0.03), Distortion=0.0, props=dict(bands_direction="X"))
                nd.set(b.inputs["Normal"], nd.node("ShaderNodeBump", Strength=0.6, Height=wv.outputs[1]).outputs[0])
            bm = bmesh.new()
            segs, sides = 160, 10
            r = gauges[i] / 2 * M
            rings = []
            for k in range(segs + 1):
                x = bx + (nut - bx) * k / segs
                rings.append([bm.verts.new((x * M, self.y(i, x) + r * math.cos(2 * math.pi * q / sides), self.z(x) + r * math.sin(2 * math.pi * q / sides)))
                              for q in range(sides)])
            for k in range(segs):
                for q in range(sides):
                    bm.faces.new((rings[k][q], rings[k][(q + 1) % sides], rings[k + 1][(q + 1) % sides], rings[k + 1][q]))
            me = bpy.data.meshes.new(f"s{i}")
            bm.to_mesh(me)
            bm.free()
            o = bpy.data.objects.new(f"s{i}", me)
            bpy.context.scene.collection.objects.link(o)
            me.materials.append(m)
            for p in me.polygons:
                p.use_smooth = True
            self.S.append(dict(me=me, base=np.array([v.co[:] for v in me.vertices])))

    def y(self, i, x):  # string i (0 = lowest) across the neck at x (cm): bass side is +y
        k = (x - self.bx) / (self.nut - self.bx)
        return ((self.n - 1) / 2 - i) * (self.sp_br * (1 - k) + self.sp_nut * k) * self.M

    def z(self, x):
        k = (x - self.bx) / (self.nut - self.bx)
        return self.z_br * (1 - k) + self.z_nut * k

    def shake(self, fi, ringing):
        """ringing: {string: (age s, fret)}."""
        M = self.M
        for i, S in enumerate(self.S):
            co = S["base"].copy()
            if i in ringing:
                age, f = ringing[i]
                amp = (0.0016 if self.n == 4 else 0.0008) * math.exp(-age / 0.5) * (1 if fi % 2 else -1)
                xs = co[:, 0] / M
                xstop = self.frets[f] if f > 0 else self.nut
                uu = np.clip((xs - self.bx) / (xstop - self.bx), 0, 1)
                co[:, 2] += amp * np.sin(np.pi * uu) * (xs <= xstop)
            S["me"].vertices.foreach_set("co", co.reshape(-1).astype(np.float32))
            S["me"].update()


def _mesh_obj(name, bm, mat, smooth_=True):
    import bpy
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = smooth_
    return o


def _pick(mat):
    """A guitar pick: a rounded triangle, thin, in its local xy plane."""
    import bpy  # noqa: F401,I001
    import bmesh
    bm = bmesh.new()
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    r = 0.0125 * (1 + 0.22 * np.cos(3 * th))
    top = [bm.verts.new((float(r_ * math.cos(a)), float(r_ * math.sin(a)), 0.0004)) for r_, a in zip(r, th)]
    bot = [bm.verts.new((float(r_ * math.cos(a)), float(r_ * math.sin(a)), -0.0004)) for r_, a in zip(r, th)]
    bm.faces.new(top)
    bm.faces.new(bot[::-1])
    for i in range(len(th)):
        bm.faces.new((top[i], bot[i], bot[(i + 1) % len(th)], top[(i + 1) % len(th)]))
    return _mesh_obj("pick", bm, mat, smooth_=False)


def _stick(mat):
    """A drumstick along local +z from its butt (0) to its tip (0.41 m): tapered, a bead at the tip."""
    import bpy  # noqa: F401,I001
    import bmesh
    bm = bmesh.new()
    sides = 16
    prof = [(0.0, 0.0072), (0.28, 0.0072), (0.37, 0.0042), (0.395, 0.0028), (0.40, 0.0045), (0.408, 0.0040), (0.41, 0.0005)]
    rings = [[bm.verts.new((r * math.cos(2 * math.pi * q / sides), r * math.sin(2 * math.pi * q / sides), z)) for q in range(sides)] for z, r in prof]
    for k in range(len(rings) - 1):
        for q in range(sides):
            bm.faces.new((rings[k][q], rings[k][(q + 1) % sides], rings[k + 1][(q + 1) % sides], rings[k + 1][q]))
    bm.faces.new(rings[0][::-1])
    return _mesh_obj("stick", bm, mat), 0.41


def _place_along(obj, a, b):
    """Put an object whose local +z runs along its length so that it starts at a and points to b."""
    from mathutils import Matrix, Vector
    q = Vector((0, 0, 1)).rotation_difference(Vector(_unit(b - a)))
    obj.matrix_world = Matrix.Translation(Vector(a)) @ q.to_matrix().to_4x4()


# ---------------------------------------------------------------- build
def build(job_path):
    import bpy
    from mathutils import Matrix, Vector

    import blendkit as bk

    job = json.load(open(job_path))
    out = os.path.dirname(job_path)
    sc = bk.scene(tuple(job["res"]), job["samples"], transparent=True)
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    mats = hand_materials(job)
    kind = job["kind"]
    t0, t1 = job["t0"], job["t1"]
    nfr = max(1, int(round((t1 - t0) * FPS)))
    M = 0.01

    if job.get("family") == "kit":
        import kit3d
        km = kit3d.model(json.load(open(kit3d.spec_path(job["model"]))), sc)
        P, groups = km["P"], km["groups"]
        hk = "hat" if "hat" in P else "hihat"
        sn, hat = groups["snare"], groups[hk]
        bpy.context.view_layer.update()
        up_s = np.array(sn.matrix_world.to_3x3() @ Vector((0, 0, 1))).copy()
        up_s /= np.linalg.norm(up_s)
        sn_top = np.array(sn.matrix_world @ Vector((0, 0, P["snare"]["depth"] / 2 * M + 0.004)))
        R_s = P["snare"]["diameter"] / 2 * M
        R_h = P[hk]["diameter"] / 2 * M
        hat_c = np.array(hat.matrix_world.translation)
        toward = np.array([0.0, 1.0, 0.0])  # the drummer sits at +y
        spot_l = sn_top + toward * (0.25 * R_s) + np.array([0.08 * R_s, 0, 0])
        spot_r = hat_c + toward * (0.55 * R_h) + np.array([0, 0, 0.006])
        hat_rot0 = hat.rotation_euler.copy()
        wood, _, wb = bk.material("hickory", (0.62, 0.46, 0.28), 0.4)
        wb.inputs["Coat Weight"].default_value = 0.5
        sticks = {s: _stick(wood) for s in ("l", "r")}
        hands = {"l": Hand3D("hand_l", "left", mats), "r": Hand3D("hand_r", "right", mats)}
        hits = [tuple(h) for h in job["notes"]]
        side_hits = {"l": [t for t, p in hits if p in LEFT_DRUMS], "r": [t for t, p in hits if p in RIGHT_DRUMS]}
        cam_tgt = Vector((spot_l + spot_r) / 2 + np.array([0, 0.05, 0.04]))
        cam = bk.camera(cam_tgt + Vector((-0.28, 0.72, 0.66)), cam_tgt, 32, dof=(0.95, 3.2))  # over the drummer's shoulder
        lights(job["light"], cam_tgt)

        def height(ts, t, H=0.13):
            past = [x for x in ts if x <= t]
            fut = [x for x in ts if x > t]
            up = 1.0 if not past else 1 - math.exp(-(t - past[-1]) / 0.05)
            down = 1.0 if not fut else min(1.0, (fut[0] - t) / 0.11) ** 0.85
            return H * min(up, down)

        def frame_fn(fi, t):
            for s in ("l", "r"):
                spot = spot_l if s == "l" else spot_r
                h = height(side_hits[s], t)
                Tp = spot + up_s * h
                F = spot + toward * 0.29 + np.array([(-0.05 if s == "l" else 0.06), 0.0, 0.10 + 0.25 * h])
                d = _unit(Tp - F)
                stick, L = sticks[s]
                _place_along(stick, Tp - d * L, Tp)
                f = _unit(np.array([d[0], d[1], 0.0]) + np.array([0, 0, -0.25]))
                dz = _unit(np.array([0, 0, 1.0]) - f * f[2])
                lat = np.cross(dz, f) if s == "r" else np.cross(f, dz)
                hp = HandPose(F - f * 0.088 + dz * 0.028, f, lat, dz)
                targets = {nm: F - d * (0.006 + 0.017 * k) - dz * 0.011 - hp.l * 0.006 for k, nm in enumerate(NAMES)}
                hands[s].pose(hp, targets, F + dz * 0.009 + hp.l * 0.007)
            past = [x for x in side_hits["r"] if x <= t]  # the hat shivers when hit
            a = 0.0 if not past else 0.05 * math.exp(-(t - past[-1]) / 0.25) * math.cos((t - past[-1]) * 40)
            hat.rotation_euler = (hat_rot0[0] + a, hat_rot0[1], hat_rot0[2])
    else:
        import guitar3d
        sp = json.load(open(guitar3d.spec_path(job["model"])))
        g = guitar3d.model(sp, sc)
        st = Strings(sp, g)
        n, nut = g["n"], g["nut"]
        notes = [tuple(x) for x in job["notes"]]
        strums = [tuple(x) for x in job.get("strums", [])]
        fret_mid = lambda f: (nut - g["scale_l"] * (1 - 2 ** (-(f - 0.3) / 12))) if f > 0 else nut - 1.0  # noqa: E731

        def note_at(t):
            past = [nt for nt in notes if nt[0] <= t + 1e-3]
            return past[-1] if past else (notes[0] if notes else (t, 0, 3))

        def hand_fret(t):
            fr = [f for (tt, s, f) in notes if f > 0 and tt <= t + 0.08]
            p = fr[-1] if fr else next((f for (tt, s, f) in notes if f > 0), 3)
            return max(1, p - 1) if p > 1 else 1

        if kind == "fret":
            pos0 = eased(hand_fret, t0 + (t1 - t0) / 2)
            cx = fret_mid(int(round(pos0)) + 1) * M
            cam_tgt = Vector((cx, -0.01, st.z_nut + 0.01))
            cam = bk.camera((cx + 0.10, -0.17, st.z_nut + 0.30), cam_tgt, 50, dof=(0.36, 2.4))
            hand = Hand3D("hand", "left", mats)
        else:
            cx = sp["strum_x"] * M
            cam_tgt = Vector((cx, 0.0, st.z_br))
            if kind == "pick":
                cam_tgt = Vector((cx, -0.005, st.z_br + 0.035))
                cam = bk.camera((cx - 0.13, -0.32, st.z_br + 0.22), cam_tgt, 50, dof=(0.38, 2.6))
                pk_m, _, pkb = bk.material("pick", (0.02, 0.02, 0.02), 0.25)
                pkb.inputs["Coat Weight"].default_value = 0.8
                pick = _pick(pk_m)
            else:
                cam = bk.camera((cx - 0.16, -0.24, st.z_br + 0.20), cam_tgt, 50, dof=(0.36, 2.4))
            hand = Hand3D("hand", "right", mats)
        lights(job["light"], cam_tgt)

        def frame_fn(fi, t):
            nt = note_at(t)
            ringing = {}
            if kind == "pick":
                past = [s_ for s_ in strums if s_[0] <= t]
                if past:
                    ringing = {i: (t - past[-1][0], 0) for i in range(n)}
            elif nt and 0 <= t - nt[0] < 1.5:
                ringing = {nt[1]: (t - nt[0], nt[2])}
            st.shake(fi, ringing)
            if kind == "fret":
                p = eased(hand_fret, t)
                xi = lambda k: fret_mid(int(round(p)) + k) * M  # noqa: E731  index -> pinky, one fret each
                hp = HandPose((xi(1.5) - 0.015, -0.068, st.z_nut - 0.085), (0.0, 0.42, 0.91), (1.0, 0.0, 0.0), (0.0, -0.91, 0.42))
                targets = {}
                act = nt if (nt and t - nt[0] < 0.6) else None
                for k, nm in enumerate(NAMES):
                    x = xi(k)
                    pressed = act is not None and act[2] > 0 and int(round(act[2] - p)) == k
                    s_ = act[1] if pressed else min(n - 1, 1 + k // 2)
                    targets[nm] = (x, st.y(s_, x / M) - (0.0 if pressed else 0.004), st.z(x / M) + 0.004 + (0.0 if pressed else 0.014 + 0.004 * k))
                hand.pose(hp, targets, (xi(1) + 0.004, 0.004, (g["T"] - 1.2) * M - 0.009))
            elif kind == "pluck":
                plucks = [nt_ for nt_ in notes if nt_[0] <= t + 0.07]
                hp = HandPose((cx + 0.03, 0.075, st.z_br + 0.115), (-0.1, -0.5, -0.86), (1.0, -0.1, 0.0), (0.0, -0.86, 0.5))
                targets = {}
                for k, nm in enumerate(("index", "middle")):
                    mine = [nt_ for j, nt_ in enumerate(plucks) if j % 2 == k]
                    x = cx + (0.012 if k == 0 else -0.004)
                    if mine:
                        tn, s_, _ = mine[-1]
                        u = t - tn
                        ys = st.y(s_, x / M)
                        if u < 0:
                            y, z = ys - 0.007, st.z(x / M) + 0.002
                        elif u < 0.06:
                            y, z = ys + 0.012 * (u / 0.06), st.z(x / M) + 0.002
                        else:
                            r_ = min(1.0, (u - 0.06) / 0.25)
                            y, z = ys + 0.012 - 0.01 * r_, st.z(x / M) + 0.002 + 0.016 * r_
                    else:
                        y, z = st.y(1, x / M), st.z(x / M) + 0.016
                    targets[nm] = (x, y, z)
                thumb = ((sp["pickups"][0]["x"] + 1.5) * M if sp.get("pickups") else cx - 0.03, st.y(0, cx / M) + 0.006, st.z_br + 0.006)
                hand.pose(hp, targets, thumb, curls=dict(ring=0.9, pinky=1.0))
            else:  # pick: the hand sweeps across the strings, down and up
                top_y, bot_y = st.y(0, cx / M) + 0.012, st.y(n - 1, cx / M) - 0.012
                y = top_y
                for (ts, dr, acc) in strums:
                    a, b = ts - 0.035, ts + 0.05
                    s0, s1 = (top_y, bot_y) if dr == "down" else (bot_y, top_y)
                    if t >= b:
                        y = s1
                        continue
                    if t >= a:
                        y = s0 + (s1 - s0) * smooth((t - a) / (b - a))
                    elif a - t < 0.12:
                        y = y + (s0 - y) * smooth(1 - (a - t) / 0.12)
                    break
                tipP = np.array([cx, y, st.z_br - 0.001])
                curls = dict(index=1.0, middle=1.1, ring=1.2, pinky=1.3)
                frame_ = ((-0.2, -0.9, -0.38), (0.95, -0.2, -0.1), (0.1, -0.4, 0.91))

                def grip(wrist):  # where the pick sits for a wrist: between the thumb tip and the index's last joint
                    hp_ = HandPose(wrist, *frame_)
                    ch = hand.chains(hp_, {}, None, curls=curls)
                    ip, inrm = ch[0]
                    tt = ip[2] + _unit(np.cross(ip[3] - ip[2], inrm)) * 0.009
                    ch2 = hand.chains(hp_, {}, tt, curls=curls)
                    return hp_, tt, (ch2[0][0][2] + ch2[0][0][3] + 2 * ch2[4][0][3]) / 4
                w0 = tipP + np.array([0.04, 0.08, 0.13])
                _, _, m0 = grip(w0)
                hp, tt, mid = grip(w0 + (tipP + np.array([0, 0, 0.016]) - m0))  # move the hand so the pick sits on the strings
                cs = hand.pose(hp, {}, tt, curls=curls)
                tipd = _unit(tipP - mid)
                nrm = _unit(cs[4][0][3] - cs[0][0][2])
                nrm = _unit(nrm - tipd * (nrm @ tipd))
                side = np.cross(nrm, tipd)
                Rm = Matrix(((tipd[0], side[0], nrm[0]), (tipd[1], side[1], nrm[1]), (tipd[2], side[2], nrm[2])))
                pick.matrix_world = Matrix.Translation(Vector(mid + tipd * 0.008)) @ Rm.to_4x4()

    cam.rotation_euler.rotate_axis("Z", math.radians(-12 if kind in ("fret", "sticks") else 10))  # a tilted frame
    bpy.context.view_layer.update()
    cam0 = cam.matrix_world.copy()
    for fi in range(nfr):
        t = t0 + fi / FPS
        frame_fn(fi, t)
        k = fi / max(1, nfr - 1)  # a slow push in
        fwd = (cam_tgt - cam0.translation).normalized()
        cam.matrix_world = Matrix.Translation(fwd * 0.03 * k) @ cam0
        if os.environ.get("CLOSEUP_DEBUG"):
            cam.data.dof.use_dof = False
            off = os.environ["CLOSEUP_DEBUG"].split(",")
            cam.location = cam_tgt + (Vector(tuple(float(v) for v in off)) if len(off) == 3 else Vector((-0.5, -1.0, 0.7)))
            cam.rotation_euler = (cam_tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
            cam.data.lens = 30
        sc.render.filepath = f"{out}/f_{fi:04d}.png"
        bpy.ops.render.render(write_still=True)
        print("closeup frame", fi, "/", nfr, flush=True)
    json.dump(dict(n=nfr, fps=FPS, res=job["res"]), open(f"{out}/meta.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
