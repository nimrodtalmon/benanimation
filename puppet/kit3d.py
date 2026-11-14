"""A drum kit in 3D (Blender, Cycles), rendered once into layers the 2D Drummer composites:

  python3 puppet/kit3d.py build classic          -> out/kitcache/classic-<hash>/ (static.png, crash_*.png, hat_*.png, meta.json)
  kit = Kit3D("classic")                         -> layers + strike points in kit units (cached; builds if missing)

The kit is described in public/kits/<name>.json (sizes and places in cm, finishes, camera). The
layers: 'static' (every piece but the moving cymbals, with the floor shadows), and one image per
state of each moving cymbal (crash wobble angles, hi-hat heights) with everything in front of it
cut out (holdout), so stacking them gives the right occlusion. meta.json maps image pixels to kit
units (the Drummer's 2D layout: the drummer's shoulders at (403, 650), 450 units = one arm = 55 cm)
and gives each piece's head center and strike points (per hand) in kit units.
"""
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS_PER_CM = 450 / 55.0  # kit units per cm at the drummer (Pepper's arm: 450 units)
SHOULDER_UNITS = (403, 650)


def spec_path(name):
    return f"{ROOT}/public/kits/{name}.json"


def cache_dir(name):
    spec = open(spec_path(name), "rb").read()
    h = hashlib.sha1(spec + open(os.path.abspath(__file__), "rb").read()).hexdigest()[:10]
    return f"{ROOT}/out/kitcache/{name}-{h}"


# ---------------------------------------------------------------- Blender side (build)
def build(name, samples=96):
    import bpy
    import bmesh
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Euler, Matrix, Vector

    spec = json.load(open(spec_path(name)))
    out = cache_dir(name)
    os.makedirs(out, exist_ok=True)
    M = 0.01  # cm -> m
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    W_, H_ = spec["camera"]["resolution"]
    sc.render.resolution_x, sc.render.resolution_y = W_, H_
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"

    # ---- materials
    def mat(name_, base, metallic=0.0, rough=0.5, coat=0.0, coat_rough=0.03, aniso=0.0):
        m = bpy.data.materials.new(name_)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*base, 1)
        b.inputs["Metallic"].default_value = metallic
        b.inputs["Roughness"].default_value = rough
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = coat_rough
        if aniso:
            b.inputs["Anisotropic"].default_value = aniso
        return m, b

    def link(m, a, b):
        m.node_tree.links.new(a, b)

    chrome, _ = mat("chrome", (0.9, 0.9, 0.92), 1.0, 0.1)
    black, _ = mat("black", (0.015, 0.015, 0.017), 0.0, 0.45)
    rubber, _ = mat("rubber", (0.02, 0.02, 0.02), 0.0, 0.8)
    felt, _ = mat("felt", (0.25, 0.02, 0.03), 0.0, 0.95)
    # red sparkle lacquer: a deep red with tiny metallic flakes under a clear coat
    lac, lb = mat("lacquer", (0.30, 0.008, 0.02), 0.3, 0.35, coat=1.0, coat_rough=0.02)
    nt = lac.node_tree
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 900.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = 0.06
    ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
    ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    link(lac, vor.outputs["Distance"], ramp.inputs["Fac"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.30, 0.008, 0.02, 1)
    mix.inputs["B"].default_value = (1.0, 0.45, 0.45, 1)
    link(lac, ramp.outputs["Color"], mix.inputs["Factor"])
    link(lac, mix.outputs["Result"], lb.inputs["Base Color"])
    mixm = nt.nodes.new("ShaderNodeMath")
    mixm.operation = "MULTIPLY_ADD"
    mixm.inputs[1].default_value = 0.7
    mixm.inputs[2].default_value = 0.25
    link(lac, ramp.outputs["Color"], mixm.inputs[0])
    link(lac, mixm.outputs[0], lb.inputs["Metallic"])
    # coated head: off-white, a little matte texture
    head, hb = mat("head", (0.80, 0.79, 0.74), 0.0, 0.6)
    nt = head.node_tree
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 160.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06
    link(head, nz.outputs["Fac"], bump.inputs["Height"])
    link(head, bump.outputs["Normal"], hb.inputs["Normal"])
    # brass: lathed (fine rings as a bump), brushed (anisotropic around the center)
    brass, bb = mat("brass", (0.72, 0.45, 0.16), 1.0, 0.24, aniso=0.6)
    nt = brass.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "RINGS"
    wave.rings_direction = "Z"
    wave.inputs["Scale"].default_value = 60.0
    wave.inputs["Distortion"].default_value = 1.5
    link(brass, tc.outputs["Object"], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.0005
    link(brass, wave.outputs["Fac"], bump.inputs["Height"])
    link(brass, bump.outputs["Normal"], bb.inputs["Normal"])
    tan = nt.nodes.new("ShaderNodeTangent")
    tan.direction_type = "RADIAL"
    tan.axis = "Z"
    link(brass, tan.outputs["Tangent"], bb.inputs["Tangent"])
    rmix = nt.nodes.new("ShaderNodeMapRange")
    rmix.inputs["To Min"].default_value = 0.18
    rmix.inputs["To Max"].default_value = 0.32
    link(brass, wave.outputs["Fac"], rmix.inputs["Value"])
    link(brass, rmix.outputs["Result"], bb.inputs["Roughness"])

    # ---- geometry helpers (meters); every part is parented to a group empty
    def smooth(o, angle=0.6):
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        bpy.ops.object.shade_smooth_by_angle(angle=angle)
        o.select_set(False)

    def put(o, parent, m_):
        o.parent = parent
        if m_ is not None:
            o.data.materials.append(m_)
        return o

    def cyl(parent, r, depth, z=0.0, m_=chrome, verts=64, cap=True, loc=None, rot=None):
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, vertices=verts, end_fill_type="NGON" if cap else "NOTHING",
                                            location=loc or (0, 0, z), rotation=rot or (0, 0, 0))
        o = put(bpy.context.object, parent, m_)
        smooth(o)
        return o

    def rod(parent, a, b, r, m_=chrome, verts=16):
        a, b = Vector(a), Vector(b)
        d = b - a
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, vertices=verts, location=(a + b) / 2)
        o = bpy.context.object
        o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        put(o, parent, m_)
        smooth(o)
        return o

    def torus(parent, R, r, z, m_=chrome, sz=1.0):
        bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=96, minor_segments=12, location=(0, 0, z))
        o = put(bpy.context.object, parent, m_)
        o.scale = (1, 1, sz)
        smooth(o, 1.2)
        return o

    def sphere(parent, r, loc, m_=chrome):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=16, ring_count=8)
        o = put(bpy.context.object, parent, m_)
        smooth(o, 1.2)
        return o

    def group(name_, loc=(0, 0, 0), rot=(0, 0, 0)):
        e = bpy.data.objects.new(name_, None)
        sc.collection.objects.link(e)
        e.location = loc
        e.rotation_euler = Euler(rot, "XYZ")
        return e

    def drum(g, r, d, lugs, shell_m, head_m=head, bottom_m=head, hoop_m=chrome, rods=True):
        """A drum along its local z: shell, heads, hoops, lugs and tension rods. Top head at +d/2."""
        cyl(g, r, d, 0, shell_m, verts=96, cap=False)
        for sgn, hm in ((1, head_m), (-1, bottom_m)):
            cyl(g, r - 0.002, 0.002, sgn * (d / 2 + 0.003), hm, verts=96)
            torus(g, r + 0.004, 0.006, sgn * (d / 2 + 0.006), hoop_m, sz=1.7)
        for i in range(lugs):
            a = 2 * math.pi * (i + 0.5) / lugs
            x, y = math.cos(a), math.sin(a)
            lr = r + 0.011
            for sgn in ((1, -1) if d > 0.25 else (1,)):
                zc = sgn * (d / 2 - (0.06 if d > 0.25 else d / 2))
                lo = cyl(g, 0.009, 0.045, 0, chrome, verts=16, loc=(lr * x, lr * y, zc))
                lo.scale = (1, 1.5, 1)
                lo.rotation_euler = (0, 0, a)
                if rods:
                    zt = sgn * (d / 2 + 0.012)
                    rod(g, (lr * x, lr * y, zc + sgn * 0.022), (lr * x, lr * y, zt), 0.0028)
                    cyl(g, 0.0055, 0.01, 0, chrome, verts=12, loc=(lr * x, lr * y, zt + sgn * 0.004))

    def cymbal(g, R, thick=0.0016, bell=0.3, bow=0.055, cup=0.032):
        """A cymbal around its local z (lathe profile), with the felt and wing nut on top."""
        me = bpy.data.meshes.new("cym")
        bm = bmesh.new()
        prof = []
        n = 60
        for i in range(n + 1):
            r = 0.0065 + (R - 0.0065) * (i / n) ** 1.2
            z = bow * R * (1 - r / R) ** 1.4 + cup * max(0.0, 1 - (r / (bell * R)) ** 2) ** 0.75
            prof.append(bm.verts.new((r, 0, z)))
        for a_, b_ in zip(prof[:-1], prof[1:]):
            bm.edges.new((a_, b_))
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new("cymbal", me)
        sc.collection.objects.link(o)
        scw = o.modifiers.new("lathe", "SCREW")
        scw.steps, scw.render_steps, scw.axis = 128, 128, "Z"
        so = o.modifiers.new("thick", "SOLIDIFY")
        so.thickness = thick
        put(o, g, brass)
        smooth(o, 1.0)
        top = bow * R + cup
        cyl(g, 0.018, 0.012, top + 0.008, felt, verts=24)
        cyl(g, 0.006, 0.05, top + 0.0, chrome, verts=12)
        sphere(g, 0.011, (0, 0, top + 0.028), chrome).scale = (1.6, 0.6, 0.8)
        return top

    def tripod(g, h, foot=0.32, collar=0.24):
        rod(g, (0, 0, collar - 0.02), (0, 0, h), 0.0125)
        for i in range(3):
            a = 2 * math.pi * i / 3 + 0.5
            fx, fy = foot * math.cos(a), foot * math.sin(a)
            rod(g, (0, 0, collar), (fx, fy, 0.012), 0.008)
            sphere(g, 0.016, (fx, fy, 0.012), rubber).scale = (1.4, 1.4, 0.7)
        cyl(g, 0.02, 0.05, collar, chrome, verts=16)

    def world_pos(c):
        return Vector(c) * M

    P = spec["pieces"]
    groups = {}
    moving = {}  # piece -> list of objects that move (the cymbal's own parts)

    def rot_of(p):
        return (math.radians(-p.get("tilt", 0)), math.radians(p.get("roll", 0)), 0)  # -x tilt: the top faces +y (the drummer)

    for key, p in P.items():
        c = world_pos(p["center"])
        t = p["type"]
        if t == "kick":
            r, d = p["diameter"] / 2 * M, p["depth"] * M
            g = group(key, c, (math.radians(90), 0, 0))  # axis toward the camera: local +z = -y (front head faces us)
            front = bpy.data.materials.new("front")
            front.use_nodes = True
            fb = front.node_tree.nodes["Principled BSDF"]
            fb.inputs["Base Color"].default_value = (0.01, 0.01, 0.012, 1)
            fb.inputs["Roughness"].default_value = 0.35
            drum(g, r, d, p.get("lugs", 10), lac, head_m=front, hoop_m=lac, rods=True)
            sk = spec.get("kick_skull")
            if sk:  # the band's art on the front head
                img = bpy.data.images.load(f"{ROOT}/{sk}")
                pm = bpy.data.materials.new("skull")
                pm.use_nodes = True
                pb = pm.node_tree.nodes["Principled BSDF"]
                tex = pm.node_tree.nodes.new("ShaderNodeTexImage")
                tex.image = img
                pm.node_tree.links.new(tex.outputs["Color"], pb.inputs["Base Color"])
                pm.node_tree.links.new(tex.outputs["Alpha"], pb.inputs["Alpha"])
                pb.inputs["Roughness"].default_value = 0.5
                asp = img.size[0] / img.size[1]
                hgt = 1.45 * r
                bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0.04 * r, d / 2 + 0.0045))
                o = put(bpy.context.object, g, pm)
                o.scale = (hgt * asp, hgt, 1)
            # port hole, lower right (viewer's)
            cyl(g, 0.06 * 2 * r / 1.1, 0.002, d / 2 + 0.005, black, verts=48, loc=(0.45 * r, -0.5 * r, d / 2 + 0.005))
            torus(g, 0.11 * r, 0.004, 0, chrome).location = (0.45 * r, -0.5 * r, d / 2 + 0.006)
            for sgn in (-1, 1):  # spurs
                rod(g, (sgn * 0.9 * r, -0.2 * r, 0.1 * d), (sgn * 1.4 * r, -1.05 * r, 0.3 * d), 0.008)
        elif t in ("tom", "snare", "floor"):
            r, d = p["diameter"] / 2 * M, p["depth"] * M
            g = group(key, c, rot_of(p))
            shell_m = chrome if t == "snare" else lac
            drum(g, r, d, p.get("lugs", 6), shell_m)
            if t == "snare":
                bpy.ops.mesh.primitive_cube_add(size=1, location=(r + 0.02, 0, 0))
                o = put(bpy.context.object, g, chrome)
                o.scale = (0.02, 0.03, 0.05)
            if t == "floor":  # three legs from brackets on the shell down to the floor
                for i in range(3):
                    a = 2 * math.pi * i / 3 + 1.2
                    x, y = (r + 0.03) * math.cos(a), (r + 0.03) * math.sin(a)
                    cyl(g, 0.014, 0.03, 0, chrome, verts=12, loc=(x, y, 0.1 * d))
                    zf = -(c.z - 0.012) / math.cos(math.radians(p.get("tilt", 0)))
                    rod(g, (x, y, 0.1 * d), (x * 1.18, y * 1.18, zf), 0.0065)
                    sphere(g, 0.014, (x * 1.18, y * 1.18, zf), rubber).scale = (1.3, 1.3, 0.7)
            groups[key] = g
            continue
        elif t in ("crash", "ride", "hihat"):
            R = p["diameter"] / 2 * M
            if t == "hihat":
                gb = group(key + "_low", c - Vector((0, 0, 0.012)), (math.radians(-p.get("tilt", 0) * 0.3), 0, 0))
                cymbal(gb, R, bow=0.03, cup=0.02)
                gb.scale = (1, 1, -1)
                gt = group(key, c, (math.radians(-p.get("tilt", 0) * 0.3), 0, 0))
                cymbal(gt, R, bow=0.03, cup=0.02)
                stand = group(key + "_stand", (c.x, c.y, 0))
                tripod(stand, c.z - 0.03)
                rod(stand, (0, 0, c.z - 0.03), (0, 0, c.z + 0.12), 0.0045)
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.12, 0.03))
                o = put(bpy.context.object, stand, black)
                o.scale = (0.09, 0.26, 0.02)
                groups[key] = gt
                moving[key] = gt
                continue
            g = group(key, c, rot_of(p))
            cymbal(g, R, bow=0.05 if t == "crash" else 0.045, cup=0.03 if t == "crash" else 0.036)
            # boom stand: a tripod, a tube, an arm reaching to the cymbal from the side away from the camera
            base = Vector((c.x + (0.25 if c.x > 0 else -0.25), c.y + 0.15, 0))
            stand = group(key + "_stand", base)
            hb_ = c.z - 0.3
            tripod(stand, hb_)
            rod(stand, (0, 0, hb_), tuple(c - base - Vector((0, 0, 0.03))), 0.009)
            groups[key] = g
            if t == "crash":
                moving[key] = g
            continue
        groups[key] = g
    # tom mount: from the kick's top to under each rack tom
    kc = world_pos(P["kick"]["center"])
    kr = P["kick"]["diameter"] / 2 * M
    mount = group("mount")
    for key in ("tom_hi", "tom_mid"):
        if key in P:
            tc_ = world_pos(P[key]["center"])
            rod(mount, (kc.x + 0.3 * (tc_.x - kc.x), kc.y, kc.z + kr), (tc_.x, tc_.y + 0.02, tc_.z - 0.11), 0.011)
    cyl(mount, 0.03, 0.05, 0, chrome, verts=16, loc=(kc.x, kc.y, kc.z + kr + 0.01))
    # snare stand
    if "snare" in P:
        sc_ = world_pos(P["snare"]["center"])
        st = group("snare_stand", (sc_.x, sc_.y, 0))
        tripod(st, sc_.z - 0.1, foot=0.26)
        for i in range(3):
            a = 2 * math.pi * i / 3 + 0.3
            rod(st, (0, 0, sc_.z - 0.1), (0.12 * math.cos(a), 0.12 * math.sin(a), sc_.z - 0.06), 0.006)

    # ---- floor (catches the shadows), lights, camera
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 1, 0))
    floor = bpy.context.object
    floor.is_shadow_catcher = True
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.03, 0.03, 0.04, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    def area(loc, target, size, energy, color):
        bpy.ops.object.light_add(type="AREA", location=loc)
        L = bpy.context.object
        L.data.energy, L.data.size, L.data.color = energy, size, color
        L.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        return L

    area((2.2, -2.8, 3.4), (0, 0.2, 0.8), 2.0, 420, (1.0, 0.95, 0.88))   # key: front right, high
    area((-2.8, -2.0, 1.6), (0, 0.2, 0.8), 2.5, 120, (0.62, 0.72, 1.0))   # fill: left, cool
    area((0.5, 2.6, 2.8), (0, 0, 0.8), 2.0, 380, (1.0, 0.55, 0.75))      # back light: rims and cymbal sheen
    area((0, -1.5, 4.0), (0, 0.3, 0.6), 3.0, 140, (1.0, 1.0, 1.0))        # top
    cam_ = spec["camera"]
    bpy.ops.object.camera_add(location=world_pos(cam_["location"]))
    cam = bpy.context.object
    cam.data.lens = cam_["focal_mm"]
    cam.rotation_euler = (world_pos(cam_["target"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    bpy.context.view_layer.update()

    def px(v):
        q = world_to_camera_view(sc, cam, Vector(v))
        return (q.x * W_, (1 - q.y) * H_)

    # ---- map image pixels to kit units: the drummer's shoulders -> (403, 650), 55 cm arm -> 450 units
    sh = world_pos(spec["drummer_shoulders"])
    a_, b_ = px(sh), px(sh + Vector((0.1, 0, 0)))
    px_per_cm = (b_[0] - a_[0]) / 10.0
    k = UNITS_PER_CM / px_per_cm  # kit units per image pixel
    x0, y0 = SHOULDER_UNITS[0] - a_[0] * k, SHOULDER_UNITS[1] - a_[1] * k
    U = lambda v: [x0 + px(v)[0] * k, y0 + px(v)[1] * k]  # noqa: E731  world -> kit units

    # ---- head centers and strike points (per hand: 'l' = viewer's left), in kit units
    pieces = {}
    for key, p in P.items():
        g = groups.get(key)
        if g is None or key == "kick":
            continue
        mw = g.matrix_world
        t = p["type"]
        if t in ("tom", "snare", "floor"):
            top = Vector((0, 0, p["depth"] / 2 * M + 0.004))
            r = p["diameter"] / 2 * M
            spots = {s_: (s_x * r, 0.32 * r) for s_, s_x in (("l", -0.15), ("r", 0.15))}  # toward the drummer (+y), a little to the hand's side
        else:
            top = Vector((0, 0, 0.0))
            r = p["diameter"] / 2 * M
            spots = {"crash": {"l": (-0.25 * r, 0.8 * r), "r": (-0.15 * r, 0.75 * r)},
                     "ride": {"l": (-0.2 * r, 0.6 * r), "r": (0.2 * r, 0.6 * r)},
                     "hihat": {"l": (-0.3 * r, 0.6 * r), "r": (0.25 * r, 0.62 * r)}}[t]
        pieces[key] = dict(center=U(mw @ top), strike={s_: U(mw @ (top + Vector((dx, dy, 0)))) for s_, (dx, dy) in spots.items()})

    # ---- render: the static kit (moving cymbals hidden), then each cymbal state over holdouts
    def descendants(o):
        out_ = [o]
        for ch in o.children:
            out_ += descendants(ch)
        return out_

    mov_objs = {key: descendants(g) for key, g in moving.items()}
    all_mov = [o for v in mov_objs.values() for o in v]
    for o in all_mov:
        o.hide_render = True
    sc.render.filepath = f"{out}/static.png"
    bpy.ops.render.render(write_still=True)
    layers = {}
    anim = spec.get("animated", {})
    floor.hide_render = True
    for o in sc.objects:
        if o.type == "MESH" and o not in all_mov:
            o.is_holdout = True
    sc.cycles.samples = max(32, samples // 2)
    for key, objs in mov_objs.items():
        g = moving[key]
        for o in objs:
            o.hide_render = False
        base_loc, base_rot = g.location.copy(), g.rotation_euler.copy()
        if key in ("crash",):
            states = [("wobble", v) for v in anim.get(key, {}).get("wobble_deg", [0])]
        else:
            states = [("lift", v) for v in anim.get(key, {}).get("lift_cm", [0])]
        files = []
        for kind, v in states:
            g.location, g.rotation_euler = base_loc.copy(), base_rot.copy()
            if kind == "wobble":
                g.rotation_euler = (Euler(base_rot).to_matrix() @ Euler((math.radians(v), 0, 0)).to_matrix()).to_euler()
            else:
                g.location = base_loc + Vector((0, 0, v * M))
            bpy.context.view_layer.update()
            # render only the cymbal's box (faster)
            pts = [o.matrix_world @ Vector(cn) for o in objs if o.type == "MESH" for cn in o.bound_box]
            xs = [px(q)[0] for q in pts]
            ys = [px(q)[1] for q in pts]
            sc.render.use_border, sc.render.use_crop_to_border = True, False
            sc.render.border_min_x = max(0.0, (min(xs) - 20) / W_)
            sc.render.border_max_x = min(1.0, (max(xs) + 20) / W_)
            sc.render.border_min_y = max(0.0, 1 - (max(ys) + 20) / H_)
            sc.render.border_max_y = min(1.0, 1 - (min(ys) - 20) / H_)
            fn = f"{key}_{kind}_{v}.png"
            sc.render.filepath = f"{out}/{fn}"
            bpy.ops.render.render(write_still=True)
            files.append([v, fn])
        layers[key] = dict(kind=states[0][0], states=files)
        g.location, g.rotation_euler = base_loc, base_rot
        for o in objs:
            o.hide_render = True
    meta = dict(name=name, size=[W_, H_], origin=[x0, y0], units_per_px=k, pieces=pieces, layers=layers,
                floor=U(Vector((0, 0.3, 0))))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print("kit built:", out)
    return out


# ---------------------------------------------------------------- 2D side (use)
class Kit3D:
    """The rendered kit for the 2D Drummer: layers resized for a scale (screen px per kit unit),
    head centers and strike points in kit units."""

    def __init__(self, name="classic", size=1.0):
        """size: the kit's size next to the drummer (1 = true size for an arm of 55 cm; cartoon
        characters with big heads look better with a bigger kit). Scales about the shoulders,
        without re-rendering."""
        import cv2
        d = cache_dir(name)
        if not os.path.exists(f"{d}/meta.json"):
            subprocess.run([sys.executable, os.path.abspath(__file__), "build", name], check=True)
        self.dir, self.meta = d, json.load(open(f"{d}/meta.json"))
        S = np.array(SHOULDER_UNITS, float)
        sc_ = lambda q: list(S + size * (np.asarray(q, float) - S))  # noqa: E731
        m = self.meta
        m["origin"], m["units_per_px"], m["floor"] = sc_(m["origin"]), m["units_per_px"] * size, sc_(m["floor"])
        for p in m["pieces"].values():
            p["center"] = sc_(p["center"])
            p["strike"] = {k_: sc_(v) for k_, v in p["strike"].items()}
        self.cv2 = cv2
        self._cache = {}

    def strike(self, what, side):
        p = self.meta["pieces"][what]
        return np.array(p["strike"][side], float)

    def center(self, what):
        return np.array(self.meta["pieces"][what]["center"], float)

    def _img(self, fn, s):
        key = (fn, round(s, 4))
        if key not in self._cache:
            cv2 = self.cv2
            im = cv2.imread(f"{self.dir}/{fn}", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            f = s * self.meta["units_per_px"]
            im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
            a = im[..., 3:4]
            im[..., :3] *= a  # premultiplied for compositing
            self._cache[key] = im
        return self._cache[key]

    def layer(self, which, s, value=None):
        """(premultiplied BGRA image, its kit-unit origin) for 'static' or a moving cymbal at the
        state nearest to value (wobble degrees / lift cm)."""
        if which == "static":
            return self._img("static.png", s)
        st = self.meta["layers"][which]["states"]
        v, fn = min(st, key=lambda e: abs(e[0] - (value or 0.0)))
        return self._img(fn, s)

    def draw(self, frame, P, s, which="static", value=None):
        """Composite a layer into the frame. P: kit units -> screen (the Drummer's P)."""
        im = self.layer(which, s, value)
        x, y = P(*self.meta["origin"])
        x, y = int(round(x)), int(round(y))
        h, w = im.shape[:2]
        H_, W_ = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        src = im[y0 - y:y1 - y, x0 - x:x1 - x]
        reg = frame[y0:y1, x0:x1]
        reg *= 1 - src[..., 3:4]
        reg += src[..., :3]

    def alpha(self, s):
        return self.layer("static", s)[..., 3]


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], samples=int(sys.argv[3]) if len(sys.argv) > 3 else 96)
