"""A city rooftop at night in 3D (Blender, Cycles), rendered once into passes the 2D side mixes per
frame (and drawn with outlines when a drawing style is active, like everything 3D):

  python3 puppet/city3d.py build rooftop        -> out/citycache/rooftop[@style]-<hash>/
  c = City3D("rooftop"); c.frame(frame, windows=0.8, blackout=0.0, sky=img)

From public/scenery/<name>.json: the roof (tar paper, puddles, a parapet), a water tower on legs, a
stair shed with a lamp, AC units, an antenna, strings of light bulbs; around it a few hundred
buildings of random sizes whose windows are lit at random, warm or cool. Passes (linear light):
  base     everything without the windows: moon, the city's glow, the bulbs and the lamp
  win      the windows' light alone
  winid    each lit window's random number (0..1): 2D turns windows on in that order, so the city
           can light up through an intro, follow the music, black out in a break
The sky stays open (transparent): any sky (skies.py) goes behind.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blendkit as bk  # noqa: E402

BUILD = 13  # bump when the Blender side changes (renders are cached by spec + BUILD)
ROOT = bk.ROOT


def spec_of(name):
    return json.load(open(f"{ROOT}/public/scenery/{name}.json"))


def cache_dir(name):
    st = bk.style()
    d = bk.cache_dir("city", name + (f"@{st}" if st else ""), spec_of(name), BUILD)
    return d


# ---------------------------------------------------------------- Blender side
def _box(name, size, loc, mat, rot=0.0):
    import bpy
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    o.rotation_euler = (0, 0, rot)
    o.data.materials.append(mat)
    return o


def _cyl(name, r, h, loc, mat, verts=32):
    import bpy
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=loc)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    return o


def _facade(name, wspec, rng, win_val):
    """Building walls with a window grid; `win_val` (a Value node per material) switches the passes:
    0 = walls only (windows dark glass), 1 = window light only, 2 = window ids."""
    m, n, b = bk.material(name, tuple(rng.uniform(0.05, 0.16) * np.array([1.0, rng.uniform(0.85, 1.0), rng.uniform(0.75, 1.0)])), 0.85)
    tc = n.node("ShaderNodeTexCoord").outputs["Object"]
    geo = n.node("ShaderNodeNewGeometry")
    # world-scale coordinates: the object's scale undone through the Object Info node is not
    # available per axis, so windows use world position instead
    P = n.xyz(geo.outputs["Position"])
    N = n.xyz(geo.outputs["Normal"])
    horiz = n.math("ADD", n.math("MULTIPLY", n.math("ABSOLUTE", N[0]), P[1]), n.math("MULTIPLY", n.math("ABSOLUTE", N[1]), P[0]))
    ww, wh = wspec["w"], wspec["h"]
    cu, cv = n.math("DIVIDE", horiz, ww), n.math("DIVIDE", P[2], wh)
    fu, fv = n.math("FRACT", cu), n.math("FRACT", cv)
    inside = n.math("MULTIPLY", n.math("MULTIPLY", n.math("GREATER_THAN", fu, 0.22), n.math("LESS_THAN", fu, 0.78)),
                    n.math("MULTIPLY", n.math("GREATER_THAN", fv, 0.28), n.math("LESS_THAN", fv, 0.86)))
    wall = n.math("LESS_THAN", n.math("ABSOLUTE", N[2]), 0.5)  # not on roofs
    inside = n.math("MULTIPLY", inside, wall)
    cell = n.node("ShaderNodeCombineXYZ", X=n.math("FLOOR", cu), Y=n.math("FLOOR", cv), Z=n.node("ShaderNodeObjectInfo").outputs["Random"])
    rnd = n.node("ShaderNodeTexWhiteNoise", Vector=cell.outputs[0], props=dict(noise_dimensions="3D"))
    r1 = rnd.outputs["Value"]
    rc = n.xyz(rnd.outputs["Color"])
    lit = n.math("MULTIPLY", inside, n.math("LESS_THAN", r1, wspec["lit"]))
    warm = n.ramp(rc[0], [(0.35, (1.0, 0.62, 0.3)), (0.75, (1.0, 0.85, 0.6)), (0.95, (0.6, 0.8, 1.0))])
    glass = (0.01, 0.012, 0.016)
    mixc = n.node("ShaderNodeMix", Factor=inside, props=dict(data_type="RGBA"))
    bricks = n.node("ShaderNodeTexNoise", Vector=tc, Scale=40.0, Detail=6.0).outputs[0]
    n.set(mixc.inputs[6], n.ramp(bricks, [(0.3, tuple(b.inputs["Base Color"].default_value[:3])),
                                          (0.8, tuple(np.array(b.inputs["Base Color"].default_value[:3]) * 1.3))]))
    mixc.inputs[7].default_value = (*glass, 1)
    n.set(b.inputs["Base Color"], mixc.outputs[2])
    n.set(b.inputs["Roughness"], n.math("SUBTRACT", 0.85, n.math("MULTIPLY", inside, 0.75)))
    # pass switch: emission = window light (pass 1) or window id (pass 2)
    is1 = n.math("COMPARE", win_val, 1.0)
    is2 = n.math("COMPARE", win_val, 2.0)
    em_light = n.math("MULTIPLY", lit, n.math("ADD", 2.5, n.math("MULTIPLY", rc[1], 4.0)))
    n.set(b.inputs["Emission Color"], n.node("ShaderNodeMix", Factor=is2, A=warm, B=n.node("ShaderNodeCombineXYZ", X=r1, Y=r1, Z=r1).outputs[0],
                                             props=dict(data_type="RGBA")).outputs[2])
    n.set(b.inputs["Emission Strength"], n.math("ADD", n.math("MULTIPLY", is1, em_light),
                                                 n.math("MULTIPLY", is2, n.math("MULTIPLY", lit, 1.0))))
    return m


def build(name):
    import bpy  # noqa: I001
    import bmesh
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    S = spec_of(name)
    out = cache_dir(name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(S["seed"])
    cam_s = S["camera"]
    sc = bk.scene(tuple(cam_s["resolution"]), S["samples"], transparent=True)
    sc.cycles.max_bounces = 4
    if bk.style():
        bk.drawn(sc, line_px=2.6 * cam_s["resolution"][0] / 1920, color=(0.03, 0.028, 0.026), crease=135.0)
    cam = bk.camera(cam_s["location"], cam_s["target"], cam_s["lens"])
    switches = []

    # ---- the city: facades share a few materials, each with its pass switch
    mats = []
    for k in range(6):
        # build the facade, then insert the switch node into its own tree
        tmp = _facade(f"fac{k}", S["windows"], rng, 0.0)
        vnode = tmp.node_tree.nodes.new("ShaderNodeValue")
        vnode.outputs[0].default_value = 0.0
        for nd in tmp.node_tree.nodes:  # the two COMPARE nodes read the switch
            if nd.bl_idname == "ShaderNodeMath" and nd.operation == "COMPARE":
                tmp.node_tree.links.new(vnode.outputs[0], nd.inputs[0])
                nd.inputs[2].default_value = 0.1  # epsilon
        switches.append(vnode)
        mats.append(tmp)
    C = S["city"]
    street = S["street"]
    for i in range(C["n"]):
        x, y = rng.uniform(*C["x"]), rng.uniform(*C["y"])
        if abs(x) < S["roof"]["w"] / 2 + 6 and y < S["roof"]["d"] / 2 + 10:
            continue
        w, d = rng.uniform(*C["w"]), rng.uniform(*C["w"])
        h = rng.uniform(*C["h"]) * (1.0 if rng.uniform() < 0.8 else 1.6)
        _box(f"b{i}", (w, d, h), (x, y, -street + h / 2), mats[int(rng.integers(len(mats)))], rot=rng.uniform(-0.05, 0.05))
        if rng.uniform() < 0.25:  # a little roof box or a spire
            _box(f"bt{i}", (w * 0.3, d * 0.3, rng.uniform(3, 9)), (x, y, -street + h + 2), mats[0])
    # the ground far below
    gm, _, _ = bk.material("street", (0.02, 0.02, 0.025), 0.9)
    _box("street", (900, 900, 1), (0, 200, -street - 0.5), gm)

    # ---- the roof
    R = S["roof"]
    roof_m, rn, rb = bk.material("tar", (0.06, 0.06, 0.065), 0.8)
    tc = rn.node("ShaderNodeTexCoord").outputs["Object"]
    noi = rn.node("ShaderNodeTexNoise", Vector=tc, Scale=3.0, Detail=10.0)
    puddle = rn.math("GREATER_THAN", rn.node("ShaderNodeTexNoise", Vector=tc, Scale=0.8, Detail=2.0).outputs[0], 0.62)
    rn.set(rb.inputs["Base Color"], rn.ramp(noi.outputs[0], [(0.3, (0.04, 0.04, 0.045)), (0.8, (0.11, 0.105, 0.1))]))
    rn.set(rb.inputs["Roughness"], rn.math("SUBTRACT", 0.85, rn.math("MULTIPLY", puddle, 0.8)))
    _box("roof", (R["w"], R["d"], 0.4), (0, R["d"] / 2 - 3, -0.2), roof_m)
    wall_m, _, _ = bk.material("parapet", (0.22, 0.2, 0.18), 0.9)
    ph = R["parapet"]
    for (sx, sy, lx, ly) in ((R["w"], 0.3, 0, R["d"] - 3), (0.3, R["d"], -R["w"] / 2, R["d"] / 2 - 3), (0.3, R["d"], R["w"] / 2, R["d"] / 2 - 3)):
        _box("par", (sx, sy, ph), (lx, ly, ph / 2), wall_m)
    P = S["props"]
    # water tower: a wooden tank on legs with a cone roof
    wood_m, wn, wb = bk.material("staves", (0.25, 0.14, 0.07), 0.8)
    wv = wn.node("ShaderNodeTexWave", Vector=wn.node("ShaderNodeTexCoord").outputs["Object"], Scale=8.0, props=dict(bands_direction="X"))
    wn.set(wb.inputs["Base Color"], wn.ramp(wv.outputs[1], [(0.2, (0.16, 0.08, 0.04)), (0.8, (0.3, 0.18, 0.09))]))
    steel_m, _, _ = bk.material("steel", (0.25, 0.25, 0.27), 0.45, 0.8)
    tx, ty = P["water_tower"]
    for a in range(4):
        ang = a * math.pi / 2 + math.pi / 4
        _cyl("leg", 0.12, 4.0, (tx + 1.6 * math.cos(ang), ty + 1.6 * math.sin(ang), 2.0), steel_m, 8)
    _cyl("tank", 2.1, 3.6, (tx, ty, 5.8), wood_m, 40)
    bpy.ops.mesh.primitive_cone_add(vertices=40, radius1=2.3, depth=1.4, location=(tx, ty, 8.3))
    bpy.context.object.data.materials.append(steel_m)
    for z in (4.6, 5.8, 7.0):  # hoops
        bpy.ops.mesh.primitive_torus_add(major_radius=2.12, minor_radius=0.04, location=(tx, ty, z))
        bpy.context.object.data.materials.append(steel_m)
    # stair shed with a door and a lamp
    shx, shy = P["shed"]
    shed_m, _, _ = bk.material("shed", (0.3, 0.27, 0.22), 0.9)
    _box("shed", (3.2, 3.0, 2.8), (shx, shy, 1.4), shed_m)
    door_m, _, _ = bk.material("door", (0.08, 0.15, 0.2), 0.5, 0.3)
    _box("door", (1.0, 0.05, 2.1), (shx - 0.5, shy - 1.52, 1.05), door_m)
    lamp_m, ln, lb = bk.material("lamp", (1, 0.8, 0.5), 0.3)
    lb.inputs["Emission Color"].default_value = (1.0, 0.75, 0.45, 1)
    lb.inputs["Emission Strength"].default_value = 30.0
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(shx - 0.5, shy - 1.65, 2.45))
    bpy.context.object.data.materials.append(lamp_m)
    bk.light("POINT", (shx - 0.5, shy - 1.9, 2.4), 60, (1.0, 0.7, 0.4), size=0.15)
    # AC units with fan grills
    ac_m, _, _ = bk.material("ac", (0.55, 0.56, 0.55), 0.5, 0.2)
    grill_m, _, _ = bk.material("grill", (0.05, 0.05, 0.05), 0.6)
    for (ax, ay) in P["ac"]:
        _box("ac", (1.8, 1.2, 1.1), (ax, ay, 0.55), ac_m)
        _cyl("fan", 0.45, 0.04, (ax, ay, 1.12), grill_m, 24)
    # antenna mast
    anx, any_ = P["antenna"]
    _cyl("mast", 0.06, 7.0, (anx, any_, 3.5), steel_m, 8)
    for z in (5.0, 6.0, 6.8):
        _box("bar", (1.6, 0.05, 0.05), (anx, any_, z), steel_m)
    red_m, _, red_b = bk.material("red", (1, 0.1, 0.05), 0.4)
    red_b.inputs["Emission Color"].default_value = (1, 0.08, 0.04, 1)
    red_b.inputs["Emission Strength"].default_value = 40.0
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.09, location=(anx, any_, 7.05))
    bpy.context.object.data.materials.append(red_m)
    # strings of bulbs: wires sagging between two points, bulbs along them
    bulb_m, _, bulb_b = bk.material("bulb", (1, 0.85, 0.6), 0.3)
    bulb_b.inputs["Emission Color"].default_value = (1.0, 0.72, 0.38, 1)
    bulb_b.inputs["Emission Strength"].default_value = 25.0
    wire_m, _, _ = bk.material("wire", (0.02, 0.02, 0.02), 0.6)
    for (a, b) in S["string_lights"]:
        a, b = np.array(a, float), np.array(b, float)
        pts = []
        for k in range(41):
            u = k / 40
            p = a + (b - a) * u
            p[2] -= 0.9 * 4 * u * (1 - u)  # sag
            pts.append(p)
        bm = bmesh.new()
        vs = [bm.verts.new(tuple(p)) for p in pts]
        for i in range(len(vs) - 1):
            bm.edges.new((vs[i], vs[i + 1]))
        me = bpy.data.meshes.new("wire")
        bm.to_mesh(me)
        bm.free()
        wo = bpy.data.objects.new("wire", me)
        sc.collection.objects.link(wo)
        sk = wo.modifiers.new("skin", "SKIN")
        for v in me.skin_vertices[0].data:
            v.radius = (0.012, 0.012)
        me.materials.append(wire_m)
        for k in range(2, 40, 3):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=tuple(pts[k] - np.array([0, 0, 0.09])))
            bpy.context.object.data.materials.append(bulb_m)
        for k in (10, 30):
            bk.light("POINT", tuple(pts[k] - np.array([0, 0, 0.12])), 25, (1.0, 0.7, 0.38), size=0.1)
    # ---- the night: moonlight, the city's orange glow from below, a dim sky
    moon = bk.light("SUN", (0, 0, 10), 0.15, (0.6, 0.7, 1.0), size=0.02)
    moon.rotation_euler = (math.radians(55), 0, math.radians(200))
    bk.light("AREA", (0, 120, -street + 5), 4000, (1.0, 0.55, 0.25), size=300, target=(0, 0, 30))
    bk.world(0.08, (0.05, 0.07, 0.14))
    bpy.context.view_layer.update()

    def set_pass(k):
        for v in switches:
            v.outputs[0].default_value = float(k)

    set_pass(0)
    bk.render(f"{out}/base.npy")
    print("pass base", flush=True)
    # windows alone: switch every other light off, keep only the facades' emission
    lights = [o for o in sc.objects if o.type == "LIGHT"]
    for o in lights:
        o.hide_render = True
    bk.world(0.0)
    for o in sc.objects:
        if o.type == "MESH" and not any(m and m.name.startswith("fac") for m in o.data.materials):
            o.is_holdout = True
    sc.render.use_freestyle = False
    sc.cycles.samples = max(8, S["samples"] // 3)
    set_pass(1)
    bk.render(f"{out}/win.npy")
    set_pass(2)
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False
    sc.view_settings.view_transform = "Standard"
    bk.render(f"{out}/winid.npy")
    print("pass windows", flush=True)
    c = world_to_camera_view(sc, cam, Vector((0, 0, 0)))
    bk.save_meta(out, name=name, size=cam_s["resolution"], roof_center_px=[c.x * cam_s["resolution"][0], (1 - c.y) * cam_s["resolution"][1]])
    print("city built:", out)
    return out


# ---------------------------------------------------------------- 2D side
class City3D:
    def __init__(self, name="rooftop"):
        d = bk.built(cache_dir(name))
        if d is None:
            d = build(name)
        self.d = d
        ld = lambda n: np.load(f"{d}/{n}.npy").astype(np.float32)  # noqa: E731
        base = ld("base")
        self.base, self.alpha = base[..., :3], np.clip(base[..., 3:4], 0, 1)
        self.win = ld("win")[..., :3]
        self.winid = ld("winid")[..., 0]
        self.winid /= max(1e-6, float(np.percentile(self.winid[self.winid > 0], 99.5)) if (self.winid > 0).any() else 1.0)  # ids spread over 0..1
        land = self.alpha[..., 0] > 0.5
        self.win_gain = 0.3
        self.exposure = 0.14 / max(1e-5, float(np.percentile(self.base.max(-1)[land], 85)))
        import ink
        disp = bk.tonemap(self.base / np.maximum(self.alpha, 1e-3), self.exposure)[..., ::-1]
        if ink.active():  # outlines are in the render; flatten its colors once (windows are lights: leave them)
            disp = ink.flat_cached(f"{d}/base.npy", disp, self.alpha[..., 0])[..., :3]
        # everything per frame is a cheap mix of images made here once
        self.land = np.ascontiguousarray(disp, np.float32)
        self.win_disp = np.ascontiguousarray(bk.tonemap(self.win * self.win_gain, self.exposure)[..., ::-1], np.float32)
        self.winid = self.winid.astype(np.float32)
        del self.base, self.win

    def frame(self, frame, windows=1.0, blackout=0.0, sky=None, flicker=None):
        """windows: the share of windows lit (in their random order); blackout: 0..1 dims the whole city's
        light (windows and glow) for a break; sky: BGR image behind."""
        on = ((self.winid < windows) & (self.winid > 0)).astype(np.float32)[..., None]
        land = self.land * (1 - 0.55 * blackout) + self.win_disp * on * (1 - blackout)
        a = self.alpha
        if sky is None:
            frame[:] = np.minimum(land, 1.0) * a
        else:
            frame[:] = np.minimum(land, 1.0) * a + sky * (1 - a)
        return frame


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
