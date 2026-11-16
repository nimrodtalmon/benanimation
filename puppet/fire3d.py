"""A campfire in 3D (Blender, Cycles), rendered once into passes the 2D side mixes per frame:

  python3 puppet/fire3d.py build campfire       -> out/firecache/campfire-<hash>/
  f = Fire3D("campfire"); f.draw(frame, t, (x, y), scale, heat)

From public/fires/<name>.json: logs (lying and leaning, bark, charred toward the middle, glowing
cracks), a ring of stones, an ash bed with coals, and a volumetric flame (noise rising through a
tapered column, colored by a fire ramp). Passes, linear light:
  amb    the logs and stones in moonlight            (always)
  lit    the same, lit only by the fire (two lights inside the flame)   (x heat x flicker)
  ember  only the glowing coals and cracks             (x 0.3 + 0.7 heat)
  flame  a seamless loop of the flame alone (logs and stones as holdouts), stored tone-mapped
The 2D side sums the light passes, tone-maps, puts them over the picture and screens the flame
on top, scaled in height by the heat.
"""
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blendkit as bk  # noqa: E402

ROOT = bk.ROOT


def spec_of(name):
    return json.load(open(f"{ROOT}/public/fires/{name}.json"))


def cache_dir(name):
    return bk.cache_dir("fire", name, spec_of(name), os.path.abspath(__file__), bk.__file__)


# ---------------------------------------------------------------- Blender side
def build(name):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    S = spec_of(name)
    out = cache_dir(name)
    os.makedirs(out, exist_ok=True)
    M = 0.01
    res = S["camera"]["resolution"]
    sc = bk.scene(res, S["samples"]["static"])
    rng = np.random.default_rng(3)
    embers = []  # every material's ember strength node

    # ---- materials
    def bark_mat(i):
        m, n, b = bk.material(f"bark{i}", (0.06, 0.035, 0.02), 0.85)
        geo = n.node("ShaderNodeNewGeometry")
        pos = geo.outputs["Position"]
        p = n.xyz(pos)
        d = n.math("SQRT", n.math("ADD", n.math("MULTIPLY", p[0], p[0]), n.math("MULTIPLY", p[1], p[1])))
        char = n.math("SUBTRACT", 1.0, n.math("DIVIDE", n.math("SUBTRACT", d, 0.05), 0.16), clamp=True)  # charred near the middle
        tc = n.node("ShaderNodeTexCoord").outputs["Object"]
        vor = n.node("ShaderNodeTexVoronoi", Vector=tc, Scale=45.0, props=dict(feature="DISTANCE_TO_EDGE"))
        noi = n.node("ShaderNodeTexNoise", Vector=tc, Scale=9.0, Detail=8.0, Roughness=0.6)
        grain = n.node("ShaderNodeTexWave", Vector=tc, Scale=3.0, Distortion=6.0, Detail=6.0)
        bark = n.ramp(n.math("MULTIPLY", noi.outputs[0], grain.outputs[1]), [(0.0, (0.02, 0.012, 0.008)), (0.35, (0.07, 0.045, 0.028)),
                                                                             (1.0, (0.16, 0.11, 0.07))])
        mix = n.node("ShaderNodeMix", Factor=char)
        mix.data_type = "RGBA"
        n.set(mix.inputs[6], bark)
        mix.inputs[7].default_value = (0.012, 0.011, 0.010, 1)
        n.set(b.inputs["Base Color"], mix.outputs[2])
        bump = n.node("ShaderNodeBump", Strength=0.6, Height=n.math("ADD", vor.outputs[0], n.math("MULTIPLY", noi.outputs[0], 0.5)))
        n.set(b.inputs["Normal"], bump.outputs[0])
        crack = n.math("SUBTRACT", 1.0, n.math("DIVIDE", vor.outputs[0], 0.035), clamp=True)
        glow = n.math("MULTIPLY", n.math("MULTIPLY", crack, char), n.math("POWER", noi.outputs[0], 4.0))
        val = n.node("ShaderNodeValue")
        val.outputs[0].default_value = 0.0
        embers.append(val)
        n.set(b.inputs["Emission Color"], n.ramp(glow, [(0.0, (0.0, 0.0, 0.0)), (0.3, (0.9, 0.12, 0.01)), (1.0, (1.0, 0.45, 0.06))]))
        n.set(b.inputs["Emission Strength"], n.math("MULTIPLY", n.math("MULTIPLY", glow, 30.0), val))
        return m

    def stone_mat(i):
        g = rng.uniform(0.10, 0.18)
        m, n, b = bk.material(f"stone{i}", (g, g * 0.97, g * 0.93), 0.8)
        tc = n.node("ShaderNodeTexCoord").outputs["Object"]
        noi = n.node("ShaderNodeTexNoise", Vector=tc, Scale=6.0, Detail=10.0, Roughness=0.65)
        n.set(b.inputs["Base Color"], n.ramp(noi.outputs[0], [(0.3, (g * 0.55, g * 0.55, g * 0.53)), (0.7, (g * 1.25, g * 1.2, g * 1.1))]))
        bump = n.node("ShaderNodeBump", Strength=0.35, Height=noi.outputs[0])
        n.set(b.inputs["Normal"], bump.outputs[0])
        return m

    def ash_mat():
        m, n, b = bk.material("ash", (0.1, 0.1, 0.1), 0.95)
        geo = n.node("ShaderNodeNewGeometry").outputs["Position"]
        p = n.xyz(geo)
        d = n.math("SQRT", n.math("ADD", n.math("MULTIPLY", p[0], p[0]), n.math("MULTIPLY", p[1], p[1])))
        noi = n.node("ShaderNodeTexNoise", Vector=geo, Scale=14.0, Detail=10.0)
        n.set(b.inputs["Base Color"], n.ramp(n.math("ADD", n.math("MULTIPLY", d, 1.2), n.math("MULTIPLY", noi.outputs[0], 0.4)),
                                             [(0.15, (0.03, 0.028, 0.025)), (0.45, (0.16, 0.15, 0.14)), (0.9, (0.07, 0.055, 0.04))]))
        R = S["ash_radius"] * M
        n.set(b.inputs["Alpha"], n.math("SUBTRACT", 1.0, n.math("POWER", n.math("DIVIDE", d, R), 3.0), clamp=True))
        coal = n.math("MULTIPLY", n.math("SUBTRACT", 1.0, n.math("DIVIDE", d, 0.24), clamp=True),
                      n.math("POWER", n.node("ShaderNodeTexVoronoi", Vector=geo, Scale=40.0).outputs[0], 0.6))
        val = n.node("ShaderNodeValue")
        val.outputs[0].default_value = 0.0
        embers.append(val)
        n.set(b.inputs["Emission Color"], n.ramp(coal, [(0.0, (0, 0, 0)), (0.4, (0.8, 0.1, 0.01)), (1.0, (1.0, 0.5, 0.08))]))
        n.set(b.inputs["Emission Strength"], n.math("MULTIPLY", n.math("MULTIPLY", coal, 12.0), val))
        n.set(b.inputs["Normal"], n.node("ShaderNodeBump", Strength=0.3, Height=noi.outputs[0]).outputs[0])
        return m

    solids = []
    # ---- ash bed
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=S["ash_radius"] * M, fill_type="NGON", location=(0, 0, 0.002))
    ash = bpy.context.object
    ash.data.materials.append(ash_mat())
    solids.append(ash)
    # ---- logs
    for i, (ang, lean, L, r, br) in enumerate(S["logs"]):
        a = math.radians(ang)
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=r * M, depth=L * M)
        o = bpy.context.object
        mod = o.modifiers.new("d", "DISPLACE")
        tx = bpy.data.textures.new(f"t{i}", "CLOUDS")
        tx.noise_scale = 0.08
        mod.texture, mod.strength = tx, 0.012
        bpy.ops.object.shade_smooth()
        if lean == 0:  # lying, through the middle
            o.rotation_euler = (math.pi / 2, 0, a)
            o.location = (math.cos(a) * br * M, math.sin(a) * br * M, r * M * 0.9)
        else:  # leaning in (teepee): the foot out at br, the top toward the middle
            e = math.radians(lean)
            foot = Vector((math.cos(a) * br * M, math.sin(a) * br * M, 0.02))
            dirv = Vector((-math.cos(a) * math.cos(e), -math.sin(a) * math.cos(e), math.sin(e)))
            o.location = foot + dirv * (L * M / 2)
            o.rotation_euler = dirv.to_track_quat("Z", "Y").to_euler()
        o.data.materials.append(bark_mat(i))
        solids.append(o)
    # ---- stones
    st = S["stones"]
    for i in range(st["n"]):
        a = 2 * math.pi * i / st["n"] + rng.uniform(-0.1, 0.1)
        s = rng.uniform(*st["size"]) * M
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=s / 2)
        o = bpy.context.object
        mod = o.modifiers.new("d", "DISPLACE")
        tx = bpy.data.textures.new(f"s{i}", "CLOUDS")
        tx.noise_scale = rng.uniform(0.04, 0.07)
        mod.texture, mod.strength = tx, s * 0.35
        bpy.ops.object.shade_smooth()
        o.scale = (rng.uniform(0.9, 1.3), rng.uniform(0.8, 1.1), rng.uniform(0.55, 0.8))
        o.rotation_euler = (0, 0, rng.uniform(0, 6.3))
        rr = st["ring"] * M * rng.uniform(0.95, 1.05)
        o.location = (math.cos(a) * rr, math.sin(a) * rr, s * 0.22)
        o.data.materials.append(stone_mat(i))
        solids.append(o)
    # ---- the flame: a box of emitting volume
    fl = S["flame"]
    Hf, Wf = fl["height"] * M, fl["width"] * M
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, Hf / 2 + 0.03))
    flame = bpy.context.object
    flame.scale = (Wf, Wf, Hf / 2)
    fm = bpy.data.materials.new("flame")
    fm.use_nodes = True
    n = bk.Nodes(fm.node_tree)
    n.N.remove(n.N["Principled BSDF"])
    outn = n.N["Material Output"]
    tval = n.node("ShaderNodeValue")
    tval.outputs[0].default_value = 0.0
    p = n.node("ShaderNodeTexCoord").outputs["Object"]
    px, py, pz = n.xyz(p)
    zn = n.math("MULTIPLY", n.math("ADD", pz, 1.0), 0.5)  # 0 at the base, 1 at the top
    rise = n.math("MULTIPLY", tval, 2.6)
    q = n.node("ShaderNodeCombineXYZ", X=n.math("MULTIPLY", px, 2.4), Y=n.math("MULTIPLY", py, 2.4), Z=n.math("SUBTRACT", n.math("MULTIPLY", pz, 1.1), rise))
    n1 = n.node("ShaderNodeTexNoise", Vector=q.outputs[0], W=n.math("MULTIPLY", tval, 0.9), Scale=2.2, Detail=8.0, Roughness=0.62, Distortion=0.35,
                props=dict(noise_dimensions="4D"))
    q2 = n.node("ShaderNodeCombineXYZ", X=n.math("MULTIPLY", px, 0.9), Y=n.math("MULTIPLY", py, 0.9), Z=n.math("SUBTRACT", n.math("MULTIPLY", pz, 0.6), n.math("MULTIPLY", tval, 1.6)))
    n2 = n.node("ShaderNodeTexNoise", Vector=q2.outputs[0], W=n.math("ADD", n.math("MULTIPLY", tval, 0.7), 5.0), Scale=1.2, Detail=3.0, props=dict(noise_dimensions="4D"))
    c2 = n.xyz(n2.outputs[1])
    sway = n.math("ADD", n.math("MULTIPLY", zn, 0.55), 0.04)
    dx = n.math("ADD", px, n.math("MULTIPLY", n.math("SUBTRACT", c2[0], 0.5), sway))
    dy = n.math("ADD", py, n.math("MULTIPLY", n.math("SUBTRACT", c2[1], 0.5), sway))
    r = n.math("SQRT", n.math("ADD", n.math("MULTIPLY", dx, dx), n.math("MULTIPLY", dy, dy)))
    R = n.math("MULTIPLY", n.math("POWER", n.math("SUBTRACT", 1.0, zn, clamp=True), 0.75), 0.85)
    shape = n.math("SUBTRACT", 1.0, n.math("DIVIDE", r, n.math("MAXIMUM", R, 0.001)), clamp=True)
    body = n.math("SUBTRACT", n.math("MULTIPLY", n1.outputs[0], 2.8), n.math("ADD", 1.05, n.math("MULTIPLY", zn, 0.45)))
    I = n.math("MULTIPLY", n.math("MULTIPLY", n.math("POWER", shape, 0.6), body), 3.2, clamp=True)
    I = n.math("MULTIPLY", I, n.math("SUBTRACT", 1.0, n.math("DIVIDE", n.math("SUBTRACT", 0.06, zn), 0.06), clamp=True))  # fade in over the base
    col = n.ramp(I, [(0.0, (0, 0, 0)), (0.12, (0.5, 0.05, 0.005)), (0.35, (1.0, 0.25, 0.02)), (0.7, (1.0, 0.45, 0.08)), (1.0, (1.0, 0.72, 0.35))])
    em = n.node("ShaderNodeEmission", Color=col, Strength=n.math("MULTIPLY", n.math("POWER", I, 1.8), 30.0))
    n.L.new(em.outputs[0], outn.inputs["Volume"])
    flame.data.materials.append(fm)

    # ---- lights, camera
    cam_s = S["camera"]
    cam = bk.camera([v * M for v in cam_s["location"]], [v * M for v in cam_s["target"]], cam_s["lens"])
    moon = bk.light("SUN", (0, 0, 5), 0.25, (0.55, 0.65, 1.0), size=0.02)
    moon.rotation_euler = (math.radians(50), 0, math.radians(-150))
    f1 = bk.light("POINT", (0, 0, 0.22), 40, (1.0, 0.42, 0.12), size=0.18)
    f2 = bk.light("POINT", (0.03, -0.02, 0.48), 18, (1.0, 0.55, 0.2), size=0.15)
    bpy.context.view_layer.update()

    def px_of(v):
        c = world_to_camera_view(sc, cam, Vector(v))
        return [c.x * res[0], (1 - c.y) * res[1]]

    base = px_of((0, 0, 0))
    right = px_of((0.5, 0, 0))
    meta = dict(name=name, size=res, base=base, px_per_m=(right[0] - base[0]) / 0.5, flame_top=px_of((0, 0, Hf + 0.03)))

    def set_pass(moon_on, fire_on, ember):
        moon.hide_render, f1.hide_render, f2.hide_render = not moon_on, not fire_on, not fire_on
        bk.world(0.12 if moon_on else 0.0, (0.03, 0.045, 0.09))
        for v in embers:
            v.outputs[0].default_value = ember

    flame.hide_render = True
    for nm, args in (("amb", (True, False, 0.0)), ("lit", (False, True, 0.0)), ("ember", (False, False, 1.0))):
        set_pass(*args)
        bk.render(f"{out}/{nm}.npy")
        print("pass", nm, flush=True)
    # ---- the flame loop: the solids hold it out
    set_pass(False, False, 0.0)
    flame.hide_render = False
    for o in solids:
        o.is_holdout = True
    sc.cycles.samples = S["samples"]["flame"]
    sc.cycles.volume_step_rate = 3.0
    lp = S["loop"]
    N, K = lp["frames"], lp["blend"]
    for f in range(N + K):
        tval.outputs[0].default_value = f / lp["fps"]
        bk.render(f"{out}/flame_raw_{f:03d}.npy")
        print("flame", f, flush=True)
    # seamless loop, stored tone-mapped (the flame is screened over the picture)
    for i in range(N):
        a = np.load(f"{out}/flame_raw_{i:03d}.npy").astype(np.float32)
        if i < K:
            b = np.load(f"{out}/flame_raw_{i + N:03d}.npy").astype(np.float32)
            a = a * (i / K) + b * (1 - i / K)
        img = (np.clip(bk.tonemap(a[..., :3]), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f"{out}/flame_{i:03d}.png", img[..., ::-1])
    for f in range(N + K):
        os.remove(f"{out}/flame_raw_{f:03d}.npy")
    meta["loop"] = N
    bk.save_meta(out, **meta)
    print("fire built:", out)
    return out


# ---------------------------------------------------------------- 2D side
class Fire3D:
    """Draws the rendered campfire: at screen point pos (the fire's base center), scale sc (1.0 =
    the render's own size), heat 0..1.6 (0.3: embers, 1: a good fire, more: flaring up)."""

    def __init__(self, name="campfire"):
        d = bk.built(cache_dir(name))
        if d is None:
            d = build(name)
        self.d = d
        self.meta = json.load(open(f"{d}/meta.json"))
        ld = lambda n: np.load(f"{d}/{n}.npy").astype(np.float32)  # noqa: E731
        self.amb, self.lit, self.ember = ld("amb"), ld("lit"), ld("ember")
        self.flames = [cv2.imread(f"{d}/flame_{i:03d}.png") for i in range(self.meta["loop"])]  # uint8: small in memory
        self._cache = {}

    def _scaled(self, key, img, s):
        k = (key, round(s, 3))
        if k not in self._cache:
            if len(self._cache) > 400:
                self._cache.clear()
            self._cache[k] = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        return self._cache[k]

    def draw(self, frame, t, pos, sc, heat):
        Hh, Ww = frame.shape[:2]
        heat = float(heat)
        fl = 1 + 0.08 * math.sin(13.1 * t) + 0.06 * math.sin(21.7 * t + 1) + 0.05 * math.sin(7.3 * t + 2)
        lin = self.amb[..., :3] * 0.6 + self.lit[..., :3] * (0.45 * heat * fl) + self.ember[..., :3] * (0.35 + 0.65 * min(1.4, heat))
        a = self.amb[..., 3:4]
        rgb = np.concatenate([bk.tonemap(lin)[..., ::-1], a], 2)  # BGR + alpha
        s = sc * 500.0 / self.meta["size"][0]  # at scale 1 the fire is 500 px wide
        img = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        bx, by = self.meta["base"]
        x0, y0 = int(round(pos[0] - bx * s)), int(round(pos[1] - by * s))
        _over(frame, img[..., :3], img[..., 3:4], x0, y0)
        # the flame: taller with heat, screened on
        i = int(t * 30) % self.meta["loop"]
        fs = max(0.05, 0.35 + 0.6 * min(1.5, heat))
        f = self.flames[i].astype(np.float32) / 255
        f = cv2.resize(f, None, fx=s * (0.85 + 0.15 * fs), fy=s * fs, interpolation=cv2.INTER_AREA)
        f *= float(np.clip((heat - 0.1) / 0.8, 0, 1))
        fx0, fy0 = int(round(pos[0] - bx * s * (0.85 + 0.15 * fs))), int(round(pos[1] - by * s * fs))
        _screen(frame, f, fx0, fy0)


def _clip(frame, img, x0, y0):
    H_, W_ = frame.shape[:2]
    h, w = img.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W_, x0 + w), min(H_, y0 + h)
    if xb <= xa or yb <= ya:
        return None
    return (slice(ya, yb), slice(xa, xb)), (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))


def _over(frame, rgb, a, x0, y0):
    c = _clip(frame, rgb, x0, y0)
    if c:
        (fy, fx), (iy, ix) = c
        sub = frame[fy, fx]
        aa = a[iy, ix]
        sub *= 1 - aa
        sub += rgb[iy, ix] * aa


def _screen(frame, img, x0, y0):
    c = _clip(frame, img, x0, y0)
    if c:
        (fy, fx), (iy, ix) = c
        sub = frame[fy, fx]
        sub[:] = 1 - (1 - np.clip(sub, 0, 1)) * (1 - img[iy, ix])


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
