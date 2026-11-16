"""Guitars and basses in 3D (Blender, Cycles), rendered once, seen from the front (neck to the right,
bass side up), without strings (the strings are drawn per frame, so they can ring):

  python3 puppet/guitar3d.py build lp_cherry      -> out/guitarcache/lp_cherry-<hash>/ (guitar.png, meta.json)
  g = Guitar3D("lp_cherry"); img, anchors = g.at(body_px)

An instrument is described in public/instruments/<name>.json: body outline (cm), scale length,
bridge place, pickups, knobs, strap buttons, finish (sunburst colors). anchors are in the image's
pixels, in the format of props.guitar (strum, bridge, nut, neck, spacing, n, neck_len) plus butt,
horn (strap buttons) and Lb (body length), so the players use either the same way.
"""
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX_PER_CM = 24.0


def spec_path(name):
    return f"{ROOT}/public/instruments/{name}.json"


def cache_dir(name):
    h = hashlib.sha1(open(spec_path(name), "rb").read() + open(os.path.abspath(__file__), "rb").read()).hexdigest()[:10]
    st = os.environ.get("BENANIM_STYLE")  # a drawn style renders its own outlines: its own cache
    return f"{ROOT}/out/guitarcache/{name}" + (f"@{st}" if st else "") + f"-{h}"


def catmull(pts, n=12):
    """A smooth closed curve through the points."""
    P = np.asarray(pts[:-1] if np.allclose(pts[0], pts[-1]) else pts, float)
    out = []
    m = len(P)
    for i in range(m):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for u in np.linspace(0, 1, n, endpoint=False):
            u2, u3 = u * u, u * u * u
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    return np.array(out)


def model(sp, sc):
    """Build the instrument's geometry and materials in the current Blender scene (cm in the spec,
    meters in Blender; the body's front at z = T, the neck along +x, bass side +y). Returns the
    numbers other builders need (strings, frets) -- see closeup3d.py."""
    import bpy  # noqa: I001
    import bmesh
    M = 0.01
    def mat(nm, base, metallic=0.0, rough=0.5, coat=0.0):
        m = bpy.data.materials.new(nm)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*base, 1)
        b.inputs["Metallic"].default_value = metallic
        b.inputs["Roughness"].default_value = rough
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.02
        return m, b

    chrome, _ = mat("chrome", (0.9, 0.9, 0.92), 1.0, 0.12)
    nickel, _ = mat("nickel", (0.75, 0.74, 0.7), 1.0, 0.25)
    black, _ = mat("black", (0.012, 0.012, 0.014), 0.0, 0.3, coat=0.6)
    rose, _ = mat("rosewood", (0.05, 0.025, 0.015), 0.0, 0.55)
    pearl, _ = mat("pearl", (0.85, 0.85, 0.8), 0.0, 0.2)
    cream, _ = mat("cream", (0.82, 0.74, 0.55), 0.0, 0.35)
    amber, _ = mat("amber", (0.75, 0.38, 0.06), 0.0, 0.15, coat=1.0)
    guard, _ = mat("guard", (0.85, 0.84, 0.8), 0.0, 0.35, coat=0.4)
    mahog, _ = mat("mahogany", (0.22, 0.06, 0.02), 0.0, 0.3, coat=1.0)
    if sp.get("neck_wood") == "maple":
        mahog, _ = mat("maple", (0.72, 0.45, 0.18), 0.0, 0.3, coat=1.0)
    knob_m = amber if sp.get("knobs") == "amber" else chrome
    # the body's finish: a sunburst (center -> edge) with a little flame figure, under a clear coat
    body_m, bb = mat("finish", (0.5, 0.1, 0.05), 0.0, 0.35, coat=1.0)
    nt = body_m.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    L = sp["body_length"] * M
    mp.inputs["Location"].default_value = (-0.5 * L / (0.5 * L), 0, 0)
    mp.inputs["Scale"].default_value = (1 / (0.55 * L), 1 / (0.42 * L), 1)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    # Mapping applies scale then location: move the center first via a second mapping
    mp.inputs["Location"].default_value = (-0.5 * L / (0.55 * L), 0, 0)
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    nt.links.new(mp.outputs["Vector"], grad.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cols = sp["finish"]["burst"]
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*cols[2], 1)
    els[1].position, els[1].color = 0.55, (*cols[0], 1)
    e = els.new(0.22)
    e.color = (*cols[1], 1)
    nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 6.0
    wave.inputs["Distortion"].default_value = 9.0
    wave.inputs["Detail"].default_value = 3.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    mixc = nt.nodes.new("ShaderNodeMix")
    mixc.data_type = "RGBA"
    mixc.blend_type = "MULTIPLY"
    mixc.inputs["Factor"].default_value = sp["finish"].get("figure", 0.2)
    nt.links.new(ramp.outputs["Color"], mixc.inputs["A"])
    nt.links.new(wave.outputs["Color"], mixc.inputs["B"])
    nt.links.new(mixc.outputs["Result"], bb.inputs["Base Color"])

    def obj_from_poly(nm, pts_cm, z0, z1, m_, bevel=0.0):
        me = bpy.data.meshes.new(nm)
        bm = bmesh.new()
        vs = [bm.verts.new((x * M, y * M, z0)) for x, y in pts_cm]
        f = bm.faces.new(vs)
        ext = bmesh.ops.extrude_face_region(bm, geom=[f])
        for v in [g for g in ext["geom"] if isinstance(g, bmesh.types.BMVert)]:
            v.co.z = z1
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(nm, me)
        sc.collection.objects.link(o)
        if bevel:
            bv = o.modifiers.new("bevel", "BEVEL")
            bv.width, bv.segments, bv.limit_method = bevel, 4, "ANGLE"
        o.data.materials.append(m_)
        for p in o.data.polygons:
            p.use_smooth = True
        return o

    def box(cx, cy, cz, sx, sy, sz, m_, bevel=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(cx * M, cy * M, cz * M))
        o = bpy.context.object
        o.scale = (sx * M, sy * M, sz * M)
        bpy.ops.object.transform_apply(scale=True)
        if bevel:
            bv = o.modifiers.new("bevel", "BEVEL")
            bv.width, bv.segments = bevel * M, 3
        o.data.materials.append(m_)
        return o

    def cyl(cx, cy, cz, r, h, m_, verts=32):
        bpy.ops.mesh.primitive_cylinder_add(radius=r * M, depth=h * M, vertices=verts, location=(cx * M, cy * M, cz * M))
        o = bpy.context.object
        o.data.materials.append(m_)
        bpy.ops.object.shade_smooth_by_angle(angle=0.6)
        return o

    # ---- body
    T = 4.5  # body thickness, cm
    outline = catmull(sp["outline"], 10)
    obj_from_poly("body", outline, 0, T * M, body_m, bevel=0.012)
    if sp.get("pickguard"):
        obj_from_poly("guard", catmull(sp["pickguard"], 6), T * M, (T + 0.15) * M, guard, bevel=0.001)
    top = T + (0.15 if sp.get("pickguard") else 0.0)
    # ---- neck, fretboard, frets, inlays
    n = sp["strings"]
    scale_l, bx = sp["scale_length"], sp["bridge_x"]
    nut = bx + scale_l
    joint = max(p[0] for p in sp["outline"] if abs(p[1]) < 4)
    w_nut, w_body = (4.3, 5.6) if n == 6 else (4.2, 6.2)
    neck0 = joint - 0.5
    fb = [(neck0, -w_body / 2), (nut, -w_nut / 2), (nut, w_nut / 2), (neck0, w_body / 2)]
    obj_from_poly("neck", fb, (T - 1.2) * M, (top + 0.3) * M, mahog, bevel=0.002)
    obj_from_poly("board", [(joint - 0.5, -w_body / 2 + 0.1), (nut, -w_nut / 2 + 0.05), (nut, w_nut / 2 - 0.05), (joint - 0.5, w_body / 2 - 0.1)],
                  (top + 0.3) * M, (top + 0.9) * M, rose)
    frets = 22 if n == 6 else 20
    width_at = lambda x: w_nut + (w_body - w_nut) * (nut - x) / (nut - (joint - 0.5))  # noqa: E731
    for k in range(1, frets + 1):
        x = nut - scale_l * (1 - 2 ** (-k / 12))
        if x < joint - 0.5:
            break
        o = cyl(x, 0, top + 0.95, 0.12, width_at(x) - 0.2, nickel, verts=8)
        o.rotation_euler = (math.pi / 2, 0, 0)
    for k in (3, 5, 7, 9, 15, 17, 19, 21, 12):
        xa = nut - scale_l * (1 - 2 ** (-(k - 1) / 12))
        xb = nut - scale_l * (1 - 2 ** (-k / 12))
        x = (xa + xb) / 2
        if x < joint:
            continue
        ys = (-0.8, 0.8) if k == 12 else (0.0,)
        for y in ys:
            cyl(x, y, top + 0.9, 0.3, 0.05, pearl, verts=16)
    box(nut + 0.25, 0, top + 0.95, 0.5, w_nut, 0.5, cream)
    # ---- headstock and tuners
    hs = sp.get("headstock", "open_book")
    if hs == "open_book":
        head = [(nut, -w_nut / 2 - 0.2), (nut + 4, -4.6), (nut + 15, -4.8), (nut + 17.5, -3.2), (nut + 18.5, -1.4), (nut + 17.6, 0), (nut + 18.5, 1.4),
                (nut + 17.5, 3.2), (nut + 15, 4.8), (nut + 4, 4.6), (nut, w_nut / 2 + 0.2)]
        obj_from_poly("head", head, (top + 0.0) * M, (top + 1.4) * M, black, bevel=0.002)
        tun = [(nut + 4.5 + 4.5 * i, sgn * 3.3) for sgn in (-1, 1) for i in range(3)]
        for x, y in tun:
            cyl(x, y, top + 1.6, 0.55, 0.5, chrome, verts=16)
            k_ = box(x, y + np.sign(y) * 2.6, top + 0.9, 1.4, 1.9, 0.6, pearl if sp.get("knobs") == "amber" else chrome, bevel=0.25)
    else:  # Fender style: the tuners in a row on the top side
        head = [(nut, -w_nut / 2), (nut + 3, -3.0), (nut + 19, -2.6), (nut + 21, -1.0), (nut + 21.5, 1.5), (nut + 20, 4.6), (nut + 14, 5.2),
                (nut + 6, 4.4), (nut + 2, w_nut / 2 + 0.6), (nut, w_nut / 2)]
        obj_from_poly("head", head, (top + 0.0) * M, (top + 1.4) * M, mahog, bevel=0.002)
        for i in range(n):
            x = nut + 4 + 4.6 * i
            cyl(x, 1.4, top + 1.7, 0.75, 0.7, chrome, verts=16)
            box(x, 4.3, top + 1.0, 2.2, 1.9, 0.6, chrome, bevel=0.3)
    # ---- pickups, bridge, tailpiece, knobs, switch, strap buttons
    for pu in sp.get("pickups", []):
        x = pu["x"]
        if pu["type"] == "humbucker":
            box(x, 0, top + 0.3, 4.6, 8.3, 0.6, cream, bevel=0.3)  # the ring
            box(x, 0, top + 0.9, 3.8, 7.0, 0.9, chrome, bevel=0.25)  # the cover
            for i in range(n):
                cyl(x - 0.8, (i - (n - 1) / 2) * 1.0, top + 1.4, 0.18, 0.1, nickel, verts=12)
        else:  # split coil (bass): two black covers, offset
            for sgn, dx in ((1, -1.5), (-1, 1.5)):
                box(x + dx, sgn * 2.0, top + 0.6, 2.6, 4.4, 1.2, black, bevel=0.5)
    if n == 6:
        box(bx, 0, top + 1.1, 1.3, 7.8, 1.0, chrome, bevel=0.3)  # tune-o-matic
        for i in range(n):
            box(bx, (i - (n - 1) / 2) * 1.05, top + 1.7, 0.6, 0.7, 0.4, chrome, bevel=0.1)
        box(bx - 5.5, 0, top + 0.9, 1.6, 8.6, 1.0, chrome, bevel=0.5)  # tailpiece
    else:
        box(bx, 0, top + 0.15, 7.0, 7.4, 0.3, chrome, bevel=0.2)  # bridge plate
        for i in range(n):
            box(bx + 0.8, (i - (n - 1) / 2) * 1.5, top + 0.8, 1.2, 1.0, 0.9, chrome, bevel=0.2)
    for x, y in sp.get("knob_xy", []):
        cyl(x, y, top + 0.9, 1.25, 1.8, knob_m, verts=24)
        cyl(x, y, top + 1.85, 1.0, 0.2, chrome if knob_m is amber else knob_m, verts=24)
    if sp.get("switch_xy"):
        x, y = sp["switch_xy"]
        cyl(x, y, top + 0.2, 1.2, 0.3, cream, verts=24)
        cyl(x, y, top + 1.4, 0.2, 2.4, chrome, verts=12)
        cyl(x, y, top + 2.7, 0.45, 0.6, cream, verts=12)
    for x, y in sp.get("strap_buttons", []):
        cyl(x + (0.6 if x < 1 else 0), y, T / 2, 0.55, 1.2, chrome, verts=16).rotation_euler = (0, math.pi / 2 if x < 1 else 0, 0)

    return dict(outline=outline, nut=nut, bx=bx, joint=joint, n=n, top=top, T=T, scale_l=scale_l, w_nut=w_nut, w_body=w_body,
                frets=[nut - scale_l * (1 - 2 ** (-k / 12)) for k in range(0, frets + 1)], mats=dict(chrome=chrome, nickel=nickel, black=black))


def build(name, samples=64):
    import bpy  # noqa: I001  (bpy first: it provides bmesh)
    import bmesh
    from mathutils import Vector

    sp = json.load(open(spec_path(name)))
    out = cache_dir(name)
    os.makedirs(out, exist_ok=True)
    M = 0.01
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"

    g = model(sp, sc)
    outline, nut, bx, joint, n = g["outline"], g["nut"], g["bx"], g["joint"], g["n"]

    # ---- camera (orthographic, from the front), lights
    xs = [p[0] for p in outline] + [nut + 22]
    ys = [p[1] for p in outline]
    x0, x1 = min(xs) - 2, max(xs) + 2
    y0, y1 = min(ys) - 2, max(ys) + 2
    W_, H_ = int((x1 - x0) * PX_PER_CM), int((y1 - y0) * PX_PER_CM)
    sc.render.resolution_x, sc.render.resolution_y = W_, H_
    if os.environ.get("BENANIM_STYLE"):  # drawn: outlines from the geometry (ink.flat() evens the colors later)
        import blendkit
        blendkit.drawn(sc, line_px=3.6 * W_ / 1600, color=(0.03, 0.028, 0.026))
    bpy.ops.object.camera_add(location=((x0 + x1) / 2 * M, (y0 + y1) / 2 * M, 2.0), rotation=(0, 0, 0))
    cam = bpy.context.object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = (x1 - x0) * M
    sc.camera = cam
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)

    def area(loc, size, energy, color=(1, 1, 1), shape="SQUARE", size_y=None):
        bpy.ops.object.light_add(type="AREA", location=loc)
        Lt = bpy.context.object
        Lt.data.energy, Lt.data.size, Lt.data.color = energy, size, color
        if size_y:
            Lt.data.shape, Lt.data.size_y = "RECTANGLE", size_y
        Lt.rotation_euler = (Vector((0.5, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        return Lt

    area((-0.6, 0.9, 1.4), 0.45, 30, (1.0, 0.95, 0.9))
    area((0.4, 1.4, 0.9), 1.0, 10, (1.0, 0.9, 0.85))  # soft side light  # key: upper left
    area((1.4, -0.8, 1.0), 1.5, 12, (0.7, 0.8, 1.0))  # fill
    area((0.25, 0.1, 1.2), 0.15, 14, (1, 1, 1), size_y=1.6)  # a strip: the long highlight on the clear coat
    sc.render.filepath = f"{out}/guitar.png"
    bpy.ops.render.render(write_still=True)

    def px(x, y):
        return [(x - x0) * PX_PER_CM, (y1 - y) * PX_PER_CM]

    sbx, sby = sp["strap_buttons"][0], sp["strap_buttons"][1]
    meta = dict(name=name, size=[W_, H_], px_per_cm=PX_PER_CM, n=n,
                anchors=dict(strum=px(sp["strum_x"], 0), bridge=px(bx, 0), nut=px(nut, 0), neck=px(joint, 0),
                             spacing=(5.2 / (n - 1) if n == 6 else 5.7 / (n - 1)) * PX_PER_CM, n=n, neck_len=(nut - joint) * PX_PER_CM,
                             butt=px(*sbx), horn=px(*sby), Lb=sp["body_length"] * PX_PER_CM))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print("guitar built:", out)


class Guitar3D:
    """The rendered instrument at a body length in screen px: (BGRA float image, anchors)."""

    def __init__(self, name):
        import cv2
        import blendkit
        d = blendkit.built(cache_dir(name))  # the newest finished render (rebuild explicitly after editing the builder)
        if d is None:
            d = cache_dir(name)
            subprocess.run([sys.executable, os.path.abspath(__file__), "build", name], check=True)
        self.meta = json.load(open(f"{d}/meta.json"))
        self.img = cv2.imread(f"{d}/guitar.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        import ink
        if ink.active():
            self.img = ink.flat_cached(f"{d}/guitar.png", self.img[..., :3], self.img[..., 3])
        self.cv2 = cv2

    def at(self, body_px):
        f = body_px / self.meta["anchors"]["Lb"]
        im = self.cv2.resize(self.img, None, fx=f, fy=f, interpolation=self.cv2.INTER_AREA)
        a = {}
        for k, v in self.meta["anchors"].items():
            a[k] = [c * f for c in v] if isinstance(v, list) else (v if k == "n" else v * f)
        return im, a


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], samples=int(sys.argv[3]) if len(sys.argv) > 3 else 64)
