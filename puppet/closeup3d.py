"""Instrument close-ups in 3D (Blender, Cycles): a player's hands on a 3D instrument, played from
the score, shallow focus. Any rig whose "play" names a 3D model (public/instruments/*.json) gets
them; the hands take the skin and sleeve colors from the rig's drawing.

  job = closeup_job(song, rig, "fret", t0, t1, light="fire")   # or "pluck"
  frames = CloseUp(job)          # renders once (cached under out/closeupcache/), then frames.at(t)

Kinds:
- "fret": the fretting hand from above the neck, fingers on the notes' strings and frets, the
  thumb behind the neck, the hand sliding along the neck between positions.
- "pluck": the plucking hand over the pickup, index and middle alternating rest strokes on the
  notes' strings, the thumb anchored, the strings shaking after each pluck.
Hands: a skeleton (wrist, palm rays, three bones a finger, the thumb) wrapped by Blender's skin
modifier and smoothed; fingers reach their targets by a small planar IK (the last joint follows
the middle one); skin with subsurface scattering, nails. Light presets: "fire" (warm from below,
cool moon rim), "sunset" (warm low side light, blue sky fill), "stage" (colored rims).
Strings: real tubes (wound ones with windings) that shake when plucked; rendered on threes
(10 fps), with a slow push-in. The 2D shot (cast/closeup_shot.py) puts it over the venue's
picture thrown out of focus.
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
FPS = 10  # on threes: steadier for the eye, a third of the renders

# finger skeleton (meters): MCP knuckle (forward, lateral toward the index side), bone lengths, radii
FINGERS = dict(index=((0.090, 0.024), (0.043, 0.026, 0.020), (0.0098, 0.0080)),
               middle=((0.094, 0.004), (0.047, 0.029, 0.021), (0.0102, 0.0083)),
               ring=((0.088, -0.015), (0.044, 0.027, 0.020), (0.0096, 0.0078)),
               pinky=((0.077, -0.032), (0.034, 0.021, 0.018), (0.0085, 0.0068)))
THUMB = ((0.022, 0.028, -0.012), (0.042, 0.032, 0.027), (0.0125, 0.0092))


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
    fx, fy, frx, fry = j["head"]["face"][:4]
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


def closeup_job(song, rig, kind, t0, t1, light="fire", res=(1280, 720), samples=24):
    j = json.load(open(f"{ROOT}/public/parts/{rig}/parts.json"))
    play = j.get("play", {})
    model = play.get("model")
    if model is None:
        return None
    inst = json.load(open(f"{ROOT}/public/instruments/{model}.json"))
    key = "bass" if play.get("instrument") == "bass" or inst["strings"] == 4 else "guitar"
    ev = song.data.get(key, [])
    notes = [(e["t"], e.get("pitch", e.get("root", 40))) for e in ev if t0 - 2.0 <= e["t"] < t1 + 0.5]
    skin, sleeve = rig_colors(rig)
    return dict(rig=rig, model=model, kind=kind, t0=round(t0, 3), t1=round(t1, 3), fps=FPS, light=light, res=list(res), samples=samples,
                notes=[[round(t, 3), s, f] for t, s, f in finger_notes(notes, inst["strings"])], skin=skin, sleeve=sleeve,
                seed=int(sum(map(ord, rig)) + 7 * int(t0)))


def job_dir(job):
    src = b"".join(open(f, "rb").read() for f in (os.path.abspath(__file__), f"{ROOT}/puppet/guitar3d.py", f"{ROOT}/public/instruments/{job['model']}.json"))
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
        """BGRA float image of the shot at time t (on twos)."""
        self.ensure()
        i = int(np.clip(round((t - self.job["t0"]) * FPS), 0, self.meta["n"] - 1))
        if i not in self.frames:
            if len(self.frames) > 40:
                self.frames.clear()
            im = cv2.imread(f"{self.d}/f_{i:04d}.png", cv2.IMREAD_UNCHANGED)
            self.frames[i] = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
        return self.frames[i]


# ---------------------------------------------------------------- Blender side
def _v(*a):
    return np.array(a, float)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def finger_ik(B, fwd, down, lens, T):
    """Planar IK: MCP and PIP flex (DIP = 0.7 PIP) so the tip reaches T. fwd/down span the plane
    (after abduction toward T). Returns the joint points [B, P1, P2, tip]."""
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
    """A hand placed by a frame (wrist, forward, lateral toward the index side, dorsal), with
    targets for fingertips; gives the skeleton's points."""

    def __init__(self, wrist, fwd, lat, dorsal):
        self.W = _v(*wrist)
        self.f, self.l, self.dz = _unit(_v(*fwd)), _unit(_v(*lat)), _unit(_v(*dorsal))

    def P(self, x, y, z=0.0):
        return self.W + self.f * x + self.l * y + self.dz * z

    def chain(self, name, target=None, curl=0.6):
        (bx, by), lens, _ = FINGERS[name]
        B = self.P(bx, by)
        if target is None:  # relaxed: curled by `curl`
            ang = [curl * 0.6, curl * 1.0, curl * 0.7]
            pts, th, p = [B], 0.0, B
            for L, a in zip(lens, ang):
                th += a
                p = p + L * (self.f * math.cos(th) - self.dz * math.sin(th))
                pts.append(p)
            return pts
        T = _v(*target)
        side = (T - B) - ((T - B) @ self.dz) * self.dz  # abduct toward the target, within reason
        fw = _unit(side) if np.linalg.norm(side) > 1e-6 else self.f
        if fw @ self.f < 0.75:
            fw = _unit(self.f * 0.75 + _unit(fw - (fw @ self.f) * self.f) * 0.66)
        down = _unit(-self.dz - ((-self.dz) @ fw) * fw)
        return finger_ik(B, fw, down, lens, T)

    def thumb(self, target=None):
        (bx, by, bz), lens, _ = THUMB
        B = self.P(bx, by, bz)
        if target is None:
            T = self.P(0.10, 0.045, -0.04)
        else:
            T = _v(*target)
        fw = _unit(T - B)
        side = _unit(np.cross(fw, self.dz) if abs(fw @ self.dz) < 0.95 else self.l)
        down = _unit(np.cross(side, fw))
        if down @ (-self.dz) < 0:
            down = -down
        return finger_ik(B, fw, down, lens, T)


def build(job_path):
    import bpy  # noqa: I001  (bpy first: it provides bmesh)
    import bmesh
    from mathutils import Matrix, Vector

    import blendkit as bk
    import guitar3d

    job = json.load(open(job_path))
    out = os.path.dirname(job_path)
    sp = json.load(open(guitar3d.spec_path(job["model"])))
    M = 0.01
    sc = bk.scene(tuple(job["res"]), job["samples"], transparent=True)
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.cycles.use_denoising = True
    g = guitar3d.model(sp, sc)
    n, top, nut, bx = g["n"], g["top"], g["nut"], g["bx"]
    frets = g["frets"]
    rng = np.random.default_rng(job["seed"])

    # ---- strings: tubes from the bridge saddles to the nut, wound ones with windings
    z_nut = (top + 1.25) * M
    z_br = (top + (1.25 if n == 4 else 1.95)) * M
    sp_br = 1.5 if n == 4 else 1.05
    sp_nut = (g["w_nut"] - 0.8) / (n - 1)

    def sy(i, x):  # string i (0 = lowest) across the neck at x (cm): bass side is +y
        k = (x - bx) / (nut - bx)
        return ((n - 1) / 2 - i) * (sp_br * (1 - k) + sp_nut * k) * M

    def sz(x):
        k = (x - bx) / (nut - bx)
        return z_br * (1 - k) + z_nut * k

    gauges = [0.105, 0.085, 0.065, 0.045] if n == 4 else [0.046, 0.036, 0.026, 0.017, 0.013, 0.010]
    strings = []
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
        rings, us = [], []
        for k in range(segs + 1):
            u = k / segs
            x = bx + (nut - bx) * u
            rings.append([bm.verts.new((x * M, sy(i, x) + r * math.cos(2 * math.pi * q / sides), sz(x) + r * math.sin(2 * math.pi * q / sides)))
                          for q in range(sides)])
            us.append(u)
        for k in range(segs):
            for q in range(sides):
                bm.faces.new((rings[k][q], rings[k][(q + 1) % sides], rings[k + 1][(q + 1) % sides], rings[k + 1][q]))
        me = bpy.data.meshes.new(f"s{i}")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(f"s{i}", me)
        sc.collection.objects.link(o)
        me.materials.append(m)
        for p in me.polygons:
            p.use_smooth = True
        base = np.array([v.co[:] for v in me.vertices])
        uu = np.repeat(np.array(us), sides)
        strings.append(dict(me=me, base=base, u=uu))

    # ---- hands: skin-modifier skeletons
    skin = np.array(job["skin"]) * 0.6
    skin_m, nd, b = bk.material("skin", tuple(skin), 0.45)
    pt = nd.node("ShaderNodeNewGeometry").outputs["Pointiness"]
    nd.set(b.inputs["Base Color"], nd.ramp(pt, [(0.46, tuple(skin)), (0.56, tuple(skin * np.array([1.08, 0.78, 0.74])))]))
    b.inputs["Subsurface Weight"].default_value = 0.35
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
    b.inputs["Subsurface Scale"].default_value = 0.01
    tcs = nd.node("ShaderNodeTexCoord").outputs["Object"]
    pores = nd.node("ShaderNodeTexNoise", Vector=tcs, Scale=900.0, Detail=4.0)
    creases = nd.node("ShaderNodeTexVoronoi", Vector=tcs, Scale=260.0, props=dict(feature="DISTANCE_TO_EDGE"))
    nd.set(b.inputs["Normal"], nd.node("ShaderNodeBump", Strength=0.12, Height=nd.math("ADD", pores.outputs[0], nd.math("MULTIPLY", creases.outputs[0], 0.6))).outputs[0])
    nail_m, _, nb_ = bk.material("nail", tuple(min(1.0, c * 1.25 + 0.06) for c in job["skin"]), 0.2)
    nb_.inputs["Coat Weight"].default_value = 0.6
    cloth_m, nd2, b2 = bk.material("sleeve", tuple(job["sleeve"]), 0.85)
    b2.inputs["Sheen Weight"].default_value = 0.6
    weave = nd2.node("ShaderNodeTexWave", Vector=nd2.node("ShaderNodeTexCoord").outputs["Object"], Scale=180.0)
    nd2.set(b2.inputs["Normal"], nd2.node("ShaderNodeBump", Strength=0.15, Height=weave.outputs[1]).outputs[0])

    names = ["index", "middle", "ring", "pinky"]

    CH = 7  # points per finger: knuckle, mid, joint, mid, joint, mid, tip

    def skeleton_edges():
        """Vertex order: wrist, forearm, thenar, palm mids x4, then each finger's 7 points, the thumb's 7."""
        edges = [(0, 1), (0, 2)]
        idx = 3
        mids = []
        for _ in names:
            edges.append((0, idx))
            mids.append(idx)
            idx += 1
        for k in range(4):
            c = list(range(idx, idx + CH))
            edges.append((mids[k], c[0]))
            edges += [(c[q], c[q + 1]) for q in range(CH - 1)]
            idx += CH
        th = list(range(idx, idx + CH))
        edges.append((2, th[0]))
        edges += [(th[q], th[q + 1]) for q in range(CH - 1)]
        idx += CH
        return edges, idx

    EDGES, NV = skeleton_edges()

    def chain_radii(r0, r1):
        J = [r0 * 1.02, r0 * 0.95, (r0 + r1) / 2 * 0.99, r1 * 0.97]  # joints a little fuller than the bones between
        out = []
        for q in range(3):
            out += [J[q], 0.9 * (J[q] + J[q + 1]) / 2]
        return out + [J[3]]

    def radii():
        K = 1.55  # the skin hull over a dense skeleton comes out thinner than its radii
        r = [0.021 * K, 0.027 * K, 0.017 * K] + [0.0145 * K] * 4
        for k in names:
            r += chain_radii(*[v * K for v in FINGERS[k][2]])
        r += chain_radii(*[v * K for v in THUMB[2]])
        return r

    RAD = radii()

    def make_hand(nm):
        me = bpy.data.meshes.new(nm)
        me.from_pydata([(0, 0, 0)] * NV, EDGES, [])
        o = bpy.data.objects.new(nm, me)
        sc.collection.objects.link(o)
        sk = o.modifiers.new("skin", "SKIN")
        sk.use_smooth_shade = True
        for i, r in enumerate(RAD):
            me.skin_vertices[0].data[i].radius = (r, r * 0.8)
        me.skin_vertices[0].data[0].use_root = True
        sub = o.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = 2
        me.materials.append(skin_m)
        nails = []
        for k in range(5):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1.0)
            nl = bpy.context.object
            nl.data.materials.append(nail_m)
            bpy.ops.object.shade_smooth()
            nails.append(nl)
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=1.0, depth=1.0)
        sl = bpy.context.object
        dm = sl.modifiers.new("d", "DISPLACE")
        tx = bpy.data.textures.new(nm + "folds", "CLOUDS")
        tx.noise_scale = 0.35
        dm.texture, dm.strength = tx, 0.12
        sl.modifiers.new("s", "SUBSURF").render_levels = 2
        sl.data.materials.append(cloth_m)
        bpy.ops.object.shade_smooth()
        return dict(o=o, me=me, nails=nails, sleeve=sl)

    def pose_hand(h, hp, targets, thumb_target=None, curls=None):
        pts = [hp.W, hp.P(-0.09, 0, -0.004), hp.P(0.03, 0.02, -0.012)]
        chains = []
        for k in names:
            pts.append(hp.P(FINGERS[k][0][0] * 0.5, FINGERS[k][0][1] * 0.8, 0.0))

        def dense(c):
            out = []
            for q in range(3):
                out += [c[q], (c[q] + c[q + 1]) / 2]
            return out + [c[3]]
        for k in names:
            c = hp.chain(k, targets.get(k), (curls or {}).get(k, 0.7))
            chains.append(c)
            pts += dense(c)
        tc = hp.thumb(thumb_target)
        chains.append(tc)
        pts += dense(tc)
        co = np.array(pts, float).reshape(-1)
        h["me"].vertices.foreach_set("co", co.astype(np.float32))
        h["me"].update()
        for nl, c, r in zip(h["nails"], chains, [FINGERS[k][2][1] * 0.95 for k in names] + [THUMB[2][1] * 0.95]):
            d = _unit(c[3] - c[2])
            up = hp.dz - (hp.dz @ d) * d
            up = _unit(up) if np.linalg.norm(up) > 1e-6 else hp.dz
            side = np.cross(up, d)
            ctr = c[3] - d * r * 0.4 + up * r * 0.74
            R = Matrix(((d[0], side[0], up[0]), (d[1], side[1], up[1]), (d[2], side[2], up[2])))
            nl.matrix_world = Matrix.Translation(Vector(ctr)) @ R.to_4x4() @ Matrix.Diagonal((r * 0.95, r * 0.72, r * 0.16, 1))
        # the sleeve: from just behind the wrist back along the forearm
        f = hp.f
        a = hp.W - f * 0.03
        bb = hp.W - f * 0.42
        d = _unit(a - bb)
        z = Vector((0, 0, 1))
        q = z.rotation_difference(Vector(d))
        h["sleeve"].matrix_world = Matrix.Translation(Vector((a + bb) / 2)) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((0.05, 0.047, 0.39, 1))

    hand = make_hand("hand")

    # ---- the notes in this shot
    notes = [tuple(x) for x in job["notes"]]
    t0, t1 = job["t0"], job["t1"]
    nfr = max(1, int(round((t1 - t0) * FPS)))
    fret_mid = lambda f: (nut - g["scale_l"] * (1 - 2 ** (-(f - 0.3) / 12))) if f > 0 else nut - 1.0  # noqa: E731  cm, just behind fret f (f may be fractional)
    Z_TIP = lambda x: sz(x) + 0.004  # noqa: E731

    def note_at(t):
        past = [nt for nt in notes if nt[0] <= t + 1e-3]
        return past[-1] if past else (notes[0] if notes else (t, 0, 3))

    def hand_fret(t):
        """Where the fretting hand sits (fret number of the index finger), eased."""
        fr = [f for (tt, s, f) in notes if f > 0 and tt <= t + 0.08]
        p = fr[-1] if fr else next((f for (tt, s, f) in notes if f > 0), 3)
        return max(1, p - 1) if p > 1 else 1

    def eased(fn, t, w=0.12):
        return float(np.mean([fn(t + d) for d in np.linspace(-w, 0.04, 7)]))

    # ---- camera, lights
    light = job["light"]
    if job["kind"] == "fret":
        pos0 = eased(hand_fret, t0 + (t1 - t0) / 2)
        cx = fret_mid(int(round(pos0)) + 1) * M
        tgt = Vector((cx, 0.0, z_nut))
        tgt = Vector((cx, -0.01, z_nut + 0.01))
        cam = bk.camera((cx + 0.10, -0.17, z_nut + 0.30), tgt, 50, dof=(0.36, 2.4))
    else:
        cx = sp["strum_x"] * M
        tgt = Vector((cx, 0.0, z_br))
        cam = bk.camera((cx - 0.16, -0.24, z_br + 0.20), tgt, 50, dof=(0.36, 2.4))
    cam.rotation_euler.rotate_axis("Z", math.radians(-12 if job["kind"] == "fret" else 10))  # a tilted, Gorillaz-like frame
    bpy.context.view_layer.update()
    cam0 = cam.matrix_world.copy()
    if light == "fire":
        bk.light("AREA", (tgt.x - 0.1, -0.6, tgt.z - 0.25), 60, (1.0, 0.45, 0.13), size=0.5, target=tgt)
        bk.light("AREA", (tgt.x + 0.2, 0.5, tgt.z + 0.6), 25, (0.55, 0.65, 1.0), size=0.6, target=tgt)
        bk.world(0.03, (0.05, 0.06, 0.12))
    elif light == "sunset":
        bk.light("AREA", (tgt.x - 0.9, -0.3, tgt.z + 0.15), 120, (1.0, 0.6, 0.3), size=0.6, target=tgt)
        bk.light("AREA", (tgt.x + 0.3, 0.4, tgt.z + 0.8), 30, (0.5, 0.62, 1.0), size=1.0, target=tgt)
        bk.world(0.15, (0.35, 0.4, 0.6))
    else:  # stage
        bk.light("AREA", (tgt.x - 0.5, 0.5, tgt.z + 0.5), 70, (1.0, 0.3, 0.6), size=0.4, target=tgt)
        bk.light("AREA", (tgt.x + 0.5, -0.5, tgt.z + 0.4), 70, (0.3, 0.5, 1.0), size=0.4, target=tgt)
        bk.light("AREA", (tgt.x, -0.4, tgt.z + 0.7), 40, (1, 1, 1), size=0.6, target=tgt)
        bk.world(0.02)

    # ---- per frame
    for fi in range(nfr):
        t = t0 + fi / FPS
        # strings: the last plucked one shakes, decaying
        nt = note_at(t)
        for i, S in enumerate(strings):
            amp = 0.0
            if nt and nt[1] == i:
                age = t - nt[0]
                if 0 <= age < 1.5:
                    amp = (0.0016 if n == 4 else 0.0008) * math.exp(-age / 0.5) * (1 if fi % 2 else -1)
            co = S["base"].copy()
            if amp:
                xs = co[:, 0] / M
                f_ = nt[2]
                xstop = frets[f_] if f_ > 0 else nut  # vibrating length: bridge -> fret
                uu = np.clip((xs - bx) / (xstop - bx), 0, 1)
                co[:, 2] += amp * np.sin(np.pi * uu) * (xs <= xstop)
            S["me"].vertices.foreach_set("co", co.reshape(-1).astype(np.float32))
            S["me"].update()
        if job["kind"] == "fret":
            p = eased(hand_fret, t)
            xi = lambda k: fret_mid(int(round(p)) + k) * M  # noqa: E731  index -> pinky, one fret each
            wrist = (xi(1.5) - 0.015, -0.068, z_nut - 0.085)
            hp = HandPose(wrist, (0.0, 0.42, 0.91), (1.0, 0.0, 0.0), (0.0, -0.91, 0.42))
            targets = {}
            act = nt if (nt and t - nt[0] < 0.6) else None
            for k, nm in enumerate(names):
                x = xi(k)
                pressed = act is not None and act[2] > 0 and int(round(act[2] - p)) == k
                s_ = act[1] if pressed else min(n - 1, 1 + k // 2)
                y = sy(s_, x / M) - (0.0 if pressed else 0.004)
                zt = Z_TIP(x / M) + (0.0 if pressed else 0.014 + 0.004 * k)
                targets[nm] = (x, y, zt)
            thumb = (xi(1) + 0.004, 0.004, (g["T"] - 1.2) * M - 0.009)
            pose_hand(hand, hp, targets, thumb)
        else:
            plucks = [nt_ for nt_ in notes if nt_[0] <= t + 0.07]
            k_last = len(plucks)
            wrist = (cx + 0.03, 0.075, z_br + 0.115)
            hp = HandPose(wrist, (-0.1, -0.5, -0.86), (1.0, -0.1, 0.0), (0.0, -0.86, 0.5))
            targets = {}
            for k, nm in enumerate(("index", "middle")):
                mine = [nt_ for j, nt_ in enumerate(plucks) if j % 2 == k]
                x = cx + (0.012 if k == 0 else -0.004)
                if mine:
                    tn, s_, _ = mine[-1]
                    u = t - tn
                    ys = sy(s_, x / M)
                    if u < 0:  # reaching in under the string (treble side)
                        y, z = ys - 0.007, sz(x / M) + 0.002
                    elif u < 0.06:  # through the string, resting on the next one toward the thumb
                        y, z = ys + 0.012 * (u / 0.06), sz(x / M) + 0.002
                    else:
                        r_ = min(1.0, (u - 0.06) / 0.25)
                        y, z = ys + 0.012 - 0.01 * r_, sz(x / M) + 0.002 + 0.016 * r_
                else:
                    y, z = sy(1, x / M), sz(x / M) + 0.016
                targets[nm] = (x, y, z)
            thumb = ((sp["pickups"][0]["x"] + 1.5) * M if sp.get("pickups") else cx - 0.03, sy(0, cx / M) + 0.006, z_br + 0.006)
            pose_hand(hand, hp, targets, thumb, curls=dict(ring=0.9, pinky=1.0))
        k = fi / max(1, nfr - 1)  # a slow push in
        fwd = (tgt - cam0.translation).normalized()
        cam.matrix_world = Matrix.Translation(fwd * 0.03 * k) @ cam0
        if os.environ.get("CLOSEUP_DEBUG"):  # a wide look at the whole setup
            cam.data.dof.use_dof = False
            cam.location = tgt + Vector((-0.5, -1.0, 0.7))
            cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
            cam.data.lens = 30
        sc.render.filepath = f"{out}/f_{fi:04d}.png"
        bpy.ops.render.render(write_still=True)
        print("closeup frame", fi, "/", nfr, flush=True)
    json.dump(dict(n=nfr, fps=FPS, res=job["res"]), open(f"{out}/meta.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
