"""Woods in 3D (Blender, Cycles), rendered once into plates and scrolling layers:

  python3 puppet/scenery3d.py build forest      -> out/scenecache/forest-<hash>/
  sc = Scenery3D("forest"); sc.walk(frame, t, travel, dusk); sc.ferns(frame, travel, dusk)
  sc.clearing(frame, heat, flicker)

From public/scenery/<name>.json. Everything procedural: conifers (a tapered trunk, whorls of
branch cards with needle-clump alpha and light coming through them), ferns (fronds of leaflet
cards), grass (hair), hills (a displaced plane far away), a physical sky (Nishita: the sun low or
just under the horizon).
- sky_sunset / sky_dusk: the sky and the far mountains (a still picture, crossfaded by dusk)
- mid, near, ferns: layers seen side-on (orthographic cameras), one period wide; they wrap, so
  they can scroll forever at their own speeds (parallax)
- clearing_amb / clearing_fire: the night clearing, moonlight only / the campfire's light only
  (linear; 2D adds them with the fire's heat and flicker)
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blendkit as bk  # noqa: E402

BUILD = 14  # bump when the Blender side changes (renders are cached by spec + BUILD)
ROOT = bk.ROOT


def spec_of(name):
    return json.load(open(f"{ROOT}/public/scenery/{name}.json"))


def cache_dir(name):
    return bk.cache_dir("scene", name, spec_of(name), BUILD)


# ---------------------------------------------------------------- Blender side
def _needle_mat(name, rng):
    m, n, b = bk.material(name, (0.02, 0.035, 0.015), 0.65)
    uv = n.node("ShaderNodeTexCoord").outputs["UV"]
    u, v, _ = n.xyz(uv)
    noi = n.node("ShaderNodeTexNoise", Vector=n.vmath("MULTIPLY", uv, (1.0, 0.6, 1.0)), Scale=10.0, Detail=4.0, Roughness=0.6)
    fur = n.node("ShaderNodeTexNoise", Vector=n.vmath("MULTIPLY", uv, (70.0, 5.0, 1.0)), Scale=1.0, Detail=2.0)  # needles across the twig
    width = n.math("MULTIPLY", n.math("POWER", n.math("SUBTRACT", 1.0, u, clamp=True), 0.45), n.math("POWER", n.math("MULTIPLY", u, 5.0, clamp=True), 0.4))
    shape = n.math("SUBTRACT", n.math("MULTIPLY", width, 0.5), n.math("ABSOLUTE", n.math("SUBTRACT", v, 0.5)))
    a = n.math("GREATER_THAN", n.math("ADD", n.math("MULTIPLY", shape, 2.2),
                                      n.math("ADD", n.math("MULTIPLY", n.math("SUBTRACT", fur.outputs[0], 0.5), 1.4), n.math("MULTIPLY", n.math("SUBTRACT", noi.outputs[0], 0.5), 0.6))), 0.12)
    g = rng.uniform(0.8, 1.2)
    col = n.ramp(noi.outputs[0], [(0.3, (0.008 * g, 0.018 * g, 0.008)), (0.7, (0.03 * g, 0.05 * g, 0.02))])
    n.set(b.inputs["Base Color"], col)
    tr = n.node("ShaderNodeBsdfTranslucent", Color=(0.12, 0.2, 0.03, 1))
    mix = n.node("ShaderNodeMixShader", Fac=0.25)
    n.L.new(b.outputs[0], mix.inputs[1])
    n.L.new(tr.outputs[0], mix.inputs[2])
    tp = n.node("ShaderNodeBsdfTransparent")
    mix2 = n.node("ShaderNodeMixShader", Fac=a)
    n.L.new(tp.outputs[0], mix2.inputs[1])
    n.L.new(mix.outputs[0], mix2.inputs[2])
    n.L.new(mix2.outputs[0], n.N["Material Output"].inputs["Surface"])
    m.blend_method = "HASHED" if hasattr(m, "blend_method") else None
    return m


def _bark_mat(name):
    m, n, b = bk.material(name, (0.05, 0.035, 0.025), 0.9)
    tc = n.node("ShaderNodeTexCoord").outputs["Object"]
    noi = n.node("ShaderNodeTexNoise", Vector=n.vmath("MULTIPLY", tc, (6.0, 6.0, 0.8)), Scale=6.0, Detail=8.0)
    n.set(b.inputs["Base Color"], n.ramp(noi.outputs[0], [(0.3, (0.02, 0.014, 0.01)), (0.8, (0.09, 0.065, 0.045))]))
    n.set(b.inputs["Normal"], n.node("ShaderNodeBump", Strength=0.8, Height=noi.outputs[0]).outputs[0])
    return m


def _leaf_mat(name, base, shape="frond"):
    """Leaf cards: a pointed leaflet shape in the card's UV, light through it."""
    m, n, b = bk.material(name, base, 0.55)
    uv = n.node("ShaderNodeTexCoord").outputs["UV"]
    u, v, _ = n.xyz(uv)
    w = n.math("MULTIPLY", n.math("POWER", n.math("SUBTRACT", 1.0, u, clamp=True), 0.6), n.math("POWER", n.math("MULTIPLY", u, 6.0, clamp=True), 0.5))
    a = n.math("GREATER_THAN", n.math("SUBTRACT", n.math("MULTIPLY", w, 0.5), n.math("ABSOLUTE", n.math("SUBTRACT", v, 0.5))), 0.0)
    noi = n.node("ShaderNodeTexNoise", Vector=uv, Scale=8.0, Detail=4.0)
    n.set(b.inputs["Base Color"], n.ramp(noi.outputs[0], [(0.3, tuple(c * 0.6 for c in base)), (0.8, tuple(c * 1.3 for c in base))]))
    tr = n.node("ShaderNodeBsdfTranslucent", Color=(base[0] * 3, base[1] * 3, base[2] * 2, 1))
    mix = n.node("ShaderNodeMixShader", Fac=0.35)
    n.L.new(b.outputs[0], mix.inputs[1])
    n.L.new(tr.outputs[0], mix.inputs[2])
    tp = n.node("ShaderNodeBsdfTransparent")
    mix2 = n.node("ShaderNodeMixShader", Fac=a)
    n.L.new(tp.outputs[0], mix2.inputs[1])
    n.L.new(mix.outputs[0], mix2.inputs[2])
    n.L.new(mix2.outputs[0], n.N["Material Output"].inputs["Surface"])
    return m


def _card(bm, uvl, P0, d, s, L, w):
    """A quad from P0 along d (length L), width w across s; u along, v across."""
    vs = [bm.verts.new(P0 - s * w / 2), bm.verts.new(P0 + s * w / 2), bm.verts.new(P0 + d * L + s * w / 2), bm.verts.new(P0 + d * L - s * w / 2)]
    f = bm.faces.new(vs)
    for lp, uv in zip(f.loops, ((0, 0), (0, 1), (1, 1), (1, 0))):
        lp[uvl].uv = uv
    return f


def conifer(name, h, rng, needle, bark):
    """A spruce-like tree as one mesh: trunk + whorls of crossed branch cards."""
    import bmesh
    import bpy
    from mathutils import Vector
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new()
    r0 = h * rng.uniform(0.012, 0.018)
    geom = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=r0, radius2=r0 * 0.08, depth=h)
    for v in geom["verts"]:
        v.co.z += h / 2
    for f in bm.faces:
        f.material_index = 1
    nlev = int(h * 3.0) + 14
    for i in range(nlev):
        rel = i / (nlev - 1)
        z = h * (0.1 + 0.88 * rel) + rng.uniform(-0.1, 0.1)
        Lb = h * (0.06 + 0.3 * (1 - rel) ** 0.85) * rng.uniform(0.8, 1.15)
        nb = int(rng.integers(6, 10))
        for k in range(nb):
            az = 2 * math.pi * (k + rng.uniform(-0.3, 0.3)) / nb + i * 0.7
            droop = math.radians(rng.uniform(5, 25) + 12 * (1 - rel))
            d = Vector((math.cos(az) * math.cos(droop), math.sin(az) * math.cos(droop), -math.sin(droop)))
            s0 = Vector((-math.sin(az), math.cos(az), 0))
            s1 = d.cross(s0).normalized()
            P0 = Vector((0, 0, z))
            _card(bm, uvl, P0, d, s0, Lb, Lb * 0.32)
            _card(bm, uvl, P0, d, s1, Lb, Lb * 0.18)
            for j in range(int(rng.integers(6, 11))):  # twigs off the branch, drooping, both sides
                uu = rng.uniform(0.2, 0.92)
                sg = 1 if j % 2 else -1
                ta = math.radians(rng.uniform(35, 65)) * sg
                td = (d * math.cos(ta) + s0 * math.sin(ta) + Vector((0, 0, -rng.uniform(0.1, 0.45)))).normalized()
                tl = Lb * 0.42 * (1 - uu) ** 0.35 * rng.uniform(0.7, 1.1)
                ts = td.cross(Vector((0, 0, 1)))
                ts = (ts.normalized() if ts.length > 1e-3 else s0)
                roll = rng.uniform(-0.6, 0.6)
                ts = (ts * math.cos(roll) + td.cross(ts) * math.sin(roll)).normalized()
                _card(bm, uvl, P0 + d * (Lb * uu), td, ts, tl, tl * 0.5)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(needle)
    me.materials.append(bark)
    return me


def fern(name, rng, mat, size=1.0):
    """A fern clump: fronds arching out, leaflet cards along each."""
    import bmesh
    import bpy
    from mathutils import Vector
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new()
    for k in range(int(rng.integers(10, 16))):
        az = rng.uniform(0, 2 * math.pi)
        L = size * rng.uniform(0.6, 1.0)
        up = math.radians(rng.uniform(40, 75))
        prev = Vector((0, 0, 0))
        n = 26
        for i in range(1, n + 1):
            s = i / n
            ang = up - s * s * math.radians(rng.uniform(60, 95))  # arches over
            hd = Vector((math.cos(az) * math.cos(ang), math.sin(az) * math.cos(ang), math.sin(ang)))
            P = prev + hd * (L / n)
            side = Vector((-math.sin(az), math.cos(az), 0))
            lw = L * 0.16 * math.sin(math.pi * min(1.0, s * 1.1)) ** 0.7 + 0.008
            for sg in (-1, 1):
                dl = (side * sg + hd * 0.35 + Vector((0, 0, -0.15))).normalized()
                _card(bm, uvl, P, dl, hd.cross(dl).normalized(), lw, lw * 0.2)
            prev = P
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    return me


def _sky(sun_elev, sun_az, strength=1.0):
    import bpy
    w = bpy.context.scene.world
    nt = w.node_tree
    sk = nt.nodes.get("Sky") or nt.nodes.new("ShaderNodeTexSky")
    sk.name = "Sky"
    sk.sky_type = "MULTIPLE_SCATTERING"
    sk.sun_elevation, sk.sun_rotation = math.radians(sun_elev), math.radians(sun_az)
    sk.sun_disc = True
    nt.links.new(sk.outputs[0], nt.nodes["Background"].inputs[0])
    nt.nodes["Background"].inputs[1].default_value = strength


def _ortho(width_m, res, z, y=-200.0, pitch=0.0):
    """Side-on camera centered on (0, 0, z), tilted down by pitch degrees."""
    import bpy
    p = math.radians(pitch)
    dist = -y
    cam = bk.camera((0, -dist * math.cos(p), z + dist * math.sin(p)), (0, 0, z), 50)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = width_m
    cam.data.clip_end = 2000
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = res
    return cam


def _instance(me, loc, rot_z=0.0, scale=1.0):
    import bpy
    o = bpy.data.objects.new(me.name + "_i", me)
    bpy.context.scene.collection.objects.link(o)
    o.location, o.rotation_euler, o.scale = loc, (0, 0, rot_z), (scale, scale, scale)
    return o


def _wrapped(xs, period):
    return [x + dx for x in xs for dx in (-period, 0, period)]


def _strip_ground(width, y0, y1, z, mat):
    import bpy
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, (y0 + y1) / 2, z))
    g = bpy.context.object
    g.scale = (width * 3, (y1 - y0), 1)
    g.data.materials.append(mat)
    return g


def _ground_mat(name, base=(0.035, 0.03, 0.02)):
    m, n, b = bk.material(name, base, 0.95)
    tc = n.node("ShaderNodeTexCoord").outputs["Object"]
    noi = n.node("ShaderNodeTexNoise", Vector=tc, Scale=40.0, Detail=10.0)
    n.set(b.inputs["Base Color"], n.ramp(noi.outputs[0], [(0.3, tuple(c * 0.5 for c in base)), (0.75, tuple(c * 1.6 for c in base))]))
    n.set(b.inputs["Normal"], n.node("ShaderNodeBump", Strength=0.5, Height=noi.outputs[0]).outputs[0])
    return m


def _grass(ground, count, length, mat):
    ps = ground.modifiers.new("grass", "PARTICLE_SYSTEM").particle_system.settings
    ps.type = "HAIR"
    ps.count = count
    ps.hair_length = length
    ps.child_type = "INTERPOLATED"
    ps.child_percent = 8
    ps.rendered_child_count = 8
    ps.root_radius, ps.tip_radius = 0.004, 0.0005
    ps.use_advanced_hair = True
    ps.brownian_factor = 0.05
    ps.material = len(ground.data.materials) + 1
    ground.data.materials.append(mat)
    return ps


def build(name, only=None):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    S = spec_of(name)
    out = cache_dir(name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    sun = S["sun"]
    meta = {}

    def fresh(samples, res, transparent=True):
        sc = bk.scene(res, samples, transparent)
        sc.cycles.transparent_max_bounces = 64
        sc.cycles.max_bounces = 6
        return sc

    def trees_lib(n, prefix):
        needle, bark = _needle_mat(prefix + "n", rng), _bark_mat(prefix + "b")
        return [conifer(f"{prefix}{i}", 1.0, rng, needle, bark) for i in range(n)]  # unit height; scaled per instance

    targets = ["far" if "far" in S else "sky", "mid", "near", "ferns", "clearing"]

    def want(k):
        return k in (only or targets)

    def mountains():
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=300, y_subdivisions=120, size=1, location=(0, 4200, 0))
        mt = bpy.context.object
        mt.scale = (14000, 3600, 1)
        mod = mt.modifiers.new("d", "DISPLACE")
        tx = bpy.data.textures.new("mt", "CLOUDS")
        tx.noise_scale, tx.noise_depth = 0.25, 4
        mod.texture, mod.strength, mod.mid_level = tx, 900, 0.45
        m, n_, b = bk.material("mt", (0.05, 0.05, 0.06), 0.9)
        mt.data.materials.append(m)
        bpy.ops.object.shade_smooth()
        return mt

    # ---- sky + mountains (perspective), sunset and dusk
    if want("sky"):
        for tag, el in (("sunset", sun["sunset_elev"]), ("dusk", sun["dusk_elev"])):
            sc = fresh(S["sky"]["samples"], S["sky"]["resolution"], transparent=False)
            _sky(el, sun["azimuth"], 1.0)
            bk.camera((0, 0, 30), (0, 1000, 120), 35).data.clip_end = 30000
            mountains()
            bk.render(f"{out}/sky_{tag}.npy")
            print("sky", tag, flush=True)

    # ---- far mountains alone (open sky: any 2D sky goes behind), lit by the low sun
    if want("far"):
        L = S["far"]
        sc = fresh(L["samples"], L["resolution"], transparent=True)
        _sky(sun["sunset_elev"] + 4, sun["azimuth"], 1.0)
        cam = bk.camera((0, 0, 30), (0, 1000, 120), L["lens"])
        cam.data.clip_end = 30000  # the mountains are kilometers away
        mountains()
        bk.render(f"{out}/far.npy")
        print("far", flush=True)

    # ---- mid layer: a ridge of trees far off, side-on
    if want("mid"):
        L = S["mid"]
        sc = fresh(L["samples"], (L["period_px"], 1080))
        _sky(sun["sunset_elev"], sun["azimuth"], 1.0)
        P = L["m_per_screen"] * L["period_px"] / 1920.0
        Hm = L["m_per_screen"] * 1080 / 1920.0
        cam = _ortho(P, (L["period_px"], 1080), Hm / 2)
        lib = trees_lib(6, "m")
        ridge_z = lambda x: Hm * (1 - L["ridge_px"] / 1080) + 6 * math.sin(2 * math.pi * x / P * 2) + 3 * math.sin(2 * math.pi * x / P * 5 + 1)  # noqa: E731
        # the hill: a profile, extruded in depth
        import bmesh
        bm = bmesh.new()
        xs = np.linspace(-1.5 * P, 1.5 * P, 600)
        top = [bm.verts.new((x, -5, ridge_z(x))) for x in xs]
        bot = [bm.verts.new((x, -5, -50)) for x in xs]
        for i in range(len(xs) - 1):
            bm.faces.new((bot[i], bot[i + 1], top[i + 1], top[i]))
        me = bpy.data.meshes.new("hill")
        bm.to_mesh(me)
        bm.free()
        hill = bpy.data.objects.new("hill", me)
        sc.collection.objects.link(hill)
        me.materials.append(_ground_mat("hillg", (0.008, 0.009, 0.006)))
        for x in _wrapped(list(np.sort(rng.uniform(-P / 2, P / 2, L["trees"]))), P):
            h = rng.uniform(*L["tree_h"])
            _instance(lib[int(rng.integers(len(lib)))], (x, rng.uniform(-4, 4), ridge_z(x) - 1.0), rng.uniform(0, 6.3), h)
        bk.render(f"{out}/mid.npy")
        print("mid", flush=True)

    # ---- near layer: big trees along the path, side-on, a bit from above
    if want("near"):
        L = S["near"]
        sc = fresh(L["samples"], (L["period_px"], 1080))
        _sky(sun["sunset_elev"], sun["azimuth"], 1.0)
        P = L["m_per_screen"] * L["period_px"] / 1920.0
        Hm = L["m_per_screen"] * 1080 / 1920.0
        gz = Hm * (1 - L["ground_px"] / 1080)
        _ortho(P, (L["period_px"], 1080), Hm / 2, pitch=0.0)
        lib = trees_lib(5, "n")
        g = _strip_ground(P, -6, 30, gz, _ground_mat("ng", (0.006, 0.006, 0.004)))
        _grass(g, 16000, 0.45, _leaf_mat("ngr", (0.02, 0.03, 0.01)))
        for x in _wrapped(list(np.sort(rng.uniform(-P / 2, P / 2, L["trees"]))), P):
            h = rng.uniform(*L["tree_h"])
            _instance(lib[int(rng.integers(len(lib)))], (x, rng.uniform(0, 12), gz), rng.uniform(0, 6.3), h)
        bk.render(f"{out}/near.npy")
        print("near", flush=True)

    # ---- ferns in front
    if want("ferns"):
        L = S["ferns"]
        sc = fresh(L["samples"], (L["period_px"], 1080))
        _sky(sun["sunset_elev"], sun["azimuth"], 1.0)
        P = L["m_per_screen"] * L["period_px"] / 1920.0
        Hm = L["m_per_screen"] * 1080 / 1920.0
        _ortho(P, (L["period_px"], 1080), Hm / 2, pitch=8.0)
        sc.render.use_border, sc.render.use_crop_to_border = True, False
        sc.render.border_min_y, sc.render.border_max_y = 0.0, 0.5
        mat = _leaf_mat("fern", (0.03, 0.06, 0.015))
        lib = [fern(f"f{i}", rng, mat) for i in range(5)]
        g = _strip_ground(P, -3, 3, -0.05, _ground_mat("fg", (0.006, 0.006, 0.004)))
        _grass(g, 2500, 0.45, _leaf_mat("fgr", (0.035, 0.05, 0.015)))
        for x in _wrapped(list(rng.uniform(-P / 2, P / 2, L["clumps"])), P):
            _instance(lib[int(rng.integers(len(lib)))], (x, rng.uniform(-1.0, 1.0), -0.05), rng.uniform(0, 6.3), rng.uniform(0.8, 1.4))
        bk.render(f"{out}/ferns.npy")
        print("ferns", flush=True)

    # ---- the clearing at night: moonlight pass and firelight pass
    if want("clearing"):
        L = S["clearing"]
        sc = fresh(L["samples"], L["resolution"], transparent=True)  # the sky stays open: any sky goes behind (skies.py)
        _sky(-14.0, sun["azimuth"] + 120, 1.0)
        cam = bk.camera(L["camera"], L["target"], L["lens"])
        lib = trees_lib(6, "c")
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 10, 0))
        g = bpy.context.object
        g.scale = (80, 60, 1)
        g.data.materials.append(_ground_mat("cg", (0.04, 0.035, 0.025)))
        _grass(g, L.get("grass", 40000), 0.14, _leaf_mat("cgr", (0.035, 0.05, 0.018)))
        cx, cy = 0.0, 3.0
        for i in range(L["trees"]):
            a = rng.uniform(-math.pi * 0.95, math.pi * 0.95) + math.pi / 2  # all around but behind the camera
            r = rng.uniform(*L["ring"])
            x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
            if y < 0 and abs(x) < 9:  # keep the view to the band clear
                continue
            if y < L["camera"][1] + 3:
                continue
            _instance(lib[int(rng.integers(len(lib)))], (x, y, 0), rng.uniform(0, 6.3), rng.uniform(*L["tree_h"]))
        moon = bk.light("SUN", (0, 0, 10), 0.06, (0.55, 0.65, 1.0), size=0.01)
        moon.rotation_euler = (math.radians(55), 0, math.radians(160))
        fire = bk.light("POINT", L["fire"], 400, (1.0, 0.45, 0.13), size=0.3)
        bpy.context.view_layer.update()
        fire.hide_render = True
        bk.render(f"{out}/clearing_amb.npy")
        fire.hide_render, moon.hide_render = False, True
        sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
        bk.render(f"{out}/clearing_fire.npy")
        c = world_to_camera_view(sc, cam, Vector(L["fire"]))
        meta["fire_px"] = [c.x * L["resolution"][0], (1 - c.y) * L["resolution"][1]]
        fire.hide_render = True
        if "sun" in L:  # a low sun (day/dawn), with the sky's light
            su = L["sun"]
            sl = bk.light("SUN", (0, 0, 10), su["energy"], (1.0, 0.82, 0.62), size=0.02)
            sl.rotation_euler = (math.radians(90 - su["elev"]), 0, math.radians(su["az"]))
            _sky(su["elev"], su["az"], 1.0)
            sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
            bk.render(f"{out}/clearing_sun.npy")
            sl.hide_render = True
            sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
        if "blaze" in L:  # the forest on fire: orange lights among the trees at the back
            bz = L["blaze"]
            spots = []
            for i in range(bz["n"]):
                a = math.pi * (0.12 + 0.76 * i / max(1, bz["n"] - 1)) + rng.uniform(-0.08, 0.08)
                r = rng.uniform(L["ring"][0] * 0.8, L["ring"][0] * 1.3)
                p = (cx + math.cos(a) * r * 1.2, cy + math.sin(a) * r, rng.uniform(0.6, 3.0))
                bk.light("POINT", p, bz["energy"] * rng.uniform(0.6, 1.3), (1.0, 0.38, 0.08), size=1.5)
                cc = world_to_camera_view(sc, cam, Vector(p))
                spots.append([cc.x * L["resolution"][0], (1 - cc.y) * L["resolution"][1]])
            meta["blaze_px"] = spots
            bk.render(f"{out}/clearing_blaze.npy")
        print("clearing", flush=True)
    old = json.load(open(f"{out}/meta.json")) if os.path.exists(f"{out}/meta.json") else {}
    old.update(meta)
    old["done"] = sorted(set(old.get("done", [])) | set(only or targets))
    bk.save_meta(out, **old)
    print("scenery built:", out)
    return out


# ---------------------------------------------------------------- 2D side
class Scenery3D:
    def __init__(self, name="forest"):
        d = bk.built(cache_dir(name))
        if d is None:
            d = build(name)
        self.d, self.S = d, spec_of(name)
        self.meta = json.load(open(f"{d}/meta.json"))
        ex = dict(self.S["exposure"])
        sky = np.load(f"{d}/sky_sunset.npy").astype(np.float32)[..., :3]
        ex["walk"] *= 0.45 / max(1e-4, float(np.median(sky.max(-1))))  # the sky's middle -> mid grey
        amb = np.load(f"{d}/clearing_amb.npy").astype(np.float32)
        land = amb[..., 3] > 0.5
        ex["night"] *= 0.05 / max(1e-5, float(np.percentile(amb[..., :3].max(-1)[land] if land.any() else amb[..., :3], 90)))

        def disp(n, e):  # linear RGBA -> display BGR + alpha, float32
            a = np.load(f"{d}/{n}.npy").astype(np.float32)
            return np.concatenate([bk.tonemap(a[..., :3], e)[..., ::-1], np.clip(a[..., 3:4], 0, 1)], 2)

        self.sky = [disp("sky_sunset", ex["walk"])[..., :3].astype(np.float16), disp("sky_dusk", ex["walk"])[..., :3].astype(np.float16)]
        self.layers = {k: disp(k, ex["walk"]).astype(np.float16) for k in ("mid", "near", "ferns")}  # half floats: half the memory
        ca = np.load(f"{d}/clearing_amb.npy").astype(np.float32)
        self.c_amb, self.c_alpha = ca[..., :3], np.clip(ca[..., 3:4], 0, 1)
        self.c_fire = np.load(f"{d}/clearing_fire.npy").astype(np.float32)[..., :3]
        self.night_exposure = ex["night"]

    def _scroll(self, frame, img, off, grade):
        Hh, Ww = frame.shape[:2]
        h, w = img.shape[:2]
        idx = (np.arange(Ww) + int(off)) % w
        lay = img[:Hh, idx].astype(np.float32)
        a = lay[..., 3:4]
        frame *= 1 - a
        frame += a * lay[..., :3] * grade

    @staticmethod
    def grade(d):
        """Sunset -> dusk: darker and bluer."""
        return np.array([1 - 0.45 * d, 1 - 0.6 * d, 1 - 0.72 * d], np.float32)

    def walk(self, frame, t, travel, dusk):
        k = float(np.clip(dusk / 0.7, 0, 1))
        frame[:] = self.sky[0].astype(np.float32) * (1 - k) + self.sky[1].astype(np.float32) * k
        g = self.grade(dusk)
        self._scroll(frame, self.layers["mid"], 0.16 * travel, g)
        self._scroll(frame, self.layers["near"], 0.55 * travel, g)
        return frame

    def ferns(self, frame, travel, dusk):
        self._scroll(frame, self.layers["ferns"], 1.5 * travel, self.grade(dusk))
        if not hasattr(self, "_foot"):  # the ground right in front: dark and out of focus
            yy = np.linspace(0, 1, frame.shape[0], dtype=np.float32)[:, None, None]
            self._foot = 1 - 0.92 * np.clip((yy - 0.80) / 0.14, 0, 1) ** 1.3
        frame *= self._foot

    def clearing(self, frame, heat, flicker=1.0, moon=1.0, sky=None):
        """The night clearing (moonlight + the fire's light) over a sky (BGR, H x W), if given."""
        lin = self.c_amb * moon + self.c_fire * (0.45 * heat * flicker)
        a = self.c_alpha
        sh = int(self.S["clearing"].get("shift", 0) * lin.shape[0])  # lower the treeline: more sky above
        if sh:
            lin = np.concatenate([np.zeros((sh,) + lin.shape[1:], np.float32), lin[:-sh]], 0)
            a = np.concatenate([np.zeros((sh,) + a.shape[1:], np.float32), a[:-sh]], 0)
        land = bk.tonemap(lin / np.maximum(a, 1e-3), self.night_exposure)[..., ::-1]
        frame[:] = land * a + (sky if sky is not None else 0.0) * (1 - a)
        return frame


class Wilds3D:
    """2D side of a scenery with far mountains and an open sky (public/scenery/wilds.json): every
    layer is drawn once in the active ink style (lines + flattened colors, cached), then each frame
    only mixes and tints. Two kinds of shot:

      walk(frame, travel, sky, tint)              # side-on: sky, far mountains, ridge, near trees
      ferns(frame, travel, tint)                  # the ferns in front of the cast (walk shots)
      clearing(frame, sky, sun=1, moon=0, fire=0, blaze=0, tint=1)   # the clearing, relit by passes

    tint: a BGR multiplier (or H x W x 3) for the light of the moment (dawn, storm, blood...)."""

    def __init__(self, name="wilds", shift=None):
        """shift: lower the clearing's treeline by this share of the height (default: the spec's)."""
        import ink
        d = bk.built(cache_dir(name))
        if d is None:
            d = build(name)
        self.d, self.S = d, spec_of(name)
        self.meta = json.load(open(f"{d}/meta.json"))
        ld = lambda n: np.load(f"{d}/{n}.npy").astype(np.float32)  # noqa: E731

        def drawn(n, key_pct=85, target=0.6, lines=True):
            a = ld(n)
            al = np.clip(a[..., 3:4], 0, 1)
            lin = a[..., :3] / np.maximum(al, 1e-3)
            land = al[..., 0] > 0.5
            e = target / max(1e-5, float(np.percentile(lin.max(-1)[land], key_pct))) if land.any() else 1.0
            disp = bk.tonemap(lin, e)[..., ::-1]
            out = ink.cached(f"{d}/{n}.npy", disp, al[..., 0], lines=lines) if ink.active() else np.concatenate([disp, al], 2)
            return np.ascontiguousarray(out, np.float32)

        self.far = drawn("far", target=0.55).astype(np.float16)
        self.layers = {k: drawn(k).astype(np.float16) for k in ("mid", "near", "ferns")}
        c = {}
        for k in ("sun", "amb", "fire", "blaze"):
            if os.path.exists(f"{d}/clearing_{k}.npy"):
                c[k] = drawn(f"clearing_{k}", key_pct=90, target={"sun": 0.75, "amb": 0.6}.get(k, 0.9), lines=False)
        self.c_alpha = np.clip(ld("clearing_amb")[..., 3:4], 0, 1)
        sh = int((self.S["clearing"].get("shift", 0) if shift is None else shift) * self.c_alpha.shape[0])  # lower the treeline
        self.shift = sh

        def down(x):
            return np.concatenate([np.zeros((sh,) + x.shape[1:], np.float32), x[:-sh]], 0) if sh else x
        self.c = {k: down(v[..., :3]) for k, v in c.items()}
        self.c_alpha = down(self.c_alpha)
        self.c_lines = None
        if ink.active():
            base = self.c.get("sun", self.c.get("amb"))
            self.c_lines = ink.Lines(base, self.c_alpha[..., 0])
        self.blaze_px = [(x, y + sh) for x, y in self.meta.get("blaze_px", [])]

    def _scroll(self, frame, img, off, tint, y=0):
        Hh, Ww = frame.shape[:2]
        h, w = img.shape[:2]
        idx = (np.arange(Ww) + int(off)) % w
        lay = img[:Hh, idx].astype(np.float32)
        if y:
            lay = np.roll(lay, y, axis=0)
            lay[:y] = 0
        a = lay[..., 3:4]
        frame *= 1 - a
        frame += a * lay[..., :3] * tint

    def walk(self, frame, travel, sky, tint=1.0, far_tint=None, horizon=0, layers=("far", "mid", "near")):
        """horizon: px to push the layers down (a lower camera: more sky). sky None: draw over frame.
        layers: which to draw (to put something between them, e.g. a giant behind the trees)."""
        if sky is not None:
            frame[:] = sky
        if "far" in layers:
            ft = tint if far_tint is None else far_tint
            h, w = self.far.shape[:2]
            x0 = int((w - frame.shape[1]) / 2 + 0.02 * travel) % max(1, w - frame.shape[1])
            lay = self.far[:frame.shape[0], x0:x0 + frame.shape[1]].astype(np.float32)
            if horizon:
                lay = np.roll(lay, horizon, axis=0)
                lay[:horizon] = 0
            a = lay[..., 3:4]
            frame *= 1 - a
            frame += a * lay[..., :3] * ft
        if "mid" in layers:
            self._scroll(frame, self.layers["mid"], 0.16 * travel, tint, horizon)
        if "near" in layers:
            self._scroll(frame, self.layers["near"], 0.55 * travel, tint, horizon)
        return frame

    def ferns(self, frame, travel, tint=1.0):
        self._scroll(frame, self.layers["ferns"], 1.5 * travel, tint)

    def clearing(self, frame, sky, sun=1.0, moon=0.0, fire=0.0, blaze=0.0, tint=1.0):
        land = np.zeros_like(self.c_alpha.repeat(3, 2))
        for k, wk in (("sun", sun), ("amb", moon), ("fire", fire), ("blaze", blaze)):
            if wk and k in self.c:
                land += self.c[k] * wk
        land *= tint
        if self.c_lines is not None:
            land = self.c_lines.draw(np.minimum(land, 1.0))
        a = self.c_alpha
        frame[:] = np.minimum(land, 1.0) * a + sky * (1 - a)
        return frame


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None)
