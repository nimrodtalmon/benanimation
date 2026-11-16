"""Small shared helpers for the Blender builders (fire3d, scenery3d, closeup3d): scene setup,
materials, node math, cameras, lights, and renders saved as linear float arrays (.npy) so 2D can
add light passes exactly before tone mapping (tonemap()).

Blender runs as a Python module (pip install bpy); builders cache their output under
out/<kind>cache/<name>-<hash>/, the hash covering the spec and the builder's own source.
"""
import hashlib
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cache_dir(kind, name, spec, build):
    """Where a builder's render lives: keyed by the spec and the builder's BUILD number (bump it when
    the Blender side changes; edits to the 2D side keep the renders)."""
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode() + str(build).encode()).hexdigest()[:10]
    return f"{ROOT}/out/{kind}cache/{name}-{h}"


def built(d):
    """d if it holds a finished build, else the newest finished build of the same kind and name
    (so editing a builder does not throw away a render; rebuild explicitly to refresh), else None."""
    import glob
    if os.path.exists(f"{d}/meta.json"):
        return d
    base = d.rsplit("-", 1)[0]
    done = [c for c in glob.glob(base + "-*") if os.path.exists(f"{c}/meta.json")]
    return max(done, key=os.path.getmtime) if done else None


def tonemap(x, exposure=1.0):
    """Linear light -> display (a soft filmic shoulder, then gamma): what the 2D side uses to show
    sums of linear passes."""
    x = np.maximum(x, 0) * exposure
    x = np.nan_to_num(x)
    L = x.max(-1, keepdims=True)  # tone-map the brightest channel, keep the hue
    Lt = 1.0 - np.exp(-L)
    c = x * (Lt / np.maximum(L, 1e-6))
    c = c + (Lt - c) * np.clip((L - 1.5) / 4.0, 0, 1) ** 2  # very bright -> toward white
    return np.clip(c, 0, 1) ** (1 / 2.2)


# ---------------------------------------------------------------- Blender side
def scene(res, samples=64, transparent=True):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.film_transparent = transparent
    sc.view_settings.view_transform = "Standard"
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGBA"
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    world(0.0)
    return sc


def style():
    """The drawing style builders should render in, from the environment (BENANIM_STYLE, set by
    ink.use() so child build processes inherit it): None (plain renders) or a style name."""
    return os.environ.get("BENANIM_STYLE") or None


def drawn(sc, line_px=2.0, color=(0.09, 0.085, 0.08), crease=125.0, wobble=1.2, seed=1):
    """Pencil/ink outlines from the geometry (Freestyle): silhouettes, borders, creases, with a
    slight wobble and pressure (thicker in the middle of a stroke). Use with ink.flat() in 2D."""
    import bpy
    sc.render.use_freestyle = True
    sc.render.line_thickness_mode = "ABSOLUTE"
    vl = bpy.context.view_layer
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    fs.crease_angle = math.radians(crease)
    ls = fs.linesets[0] if len(fs.linesets) else fs.linesets.new("ink")
    ls.select_by_visibility = True
    ls.select_silhouette = ls.select_border = ls.select_crease = True
    ls.select_external_contour = True
    if ls.linestyle is None:
        ls.linestyle = bpy.data.linestyles.new("ink")
    st = ls.linestyle
    st.color, st.thickness, st.alpha = color, line_px, 1.0
    st.thickness_position = "CENTER"
    st.caps = "ROUND"
    if wobble:
        n = st.geometry_modifiers.new("wobble", "PERLIN_NOISE_1D")
        n.frequency, n.amplitude, n.seed = 12.0, wobble, seed
    tm = st.thickness_modifiers.new("pressure", "ALONG_STROKE")
    tm.blend, tm.influence = "MULTIPLY", 0.6
    return st


def world(strength, color=(0.02, 0.03, 0.06)):
    import bpy
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*color, 1)
    bg.inputs[1].default_value = strength


def render(path, tmp=None):
    """Render the scene; save linear RGBA as float16 .npy (top row first)."""
    import bpy
    sc = bpy.context.scene
    tmp = tmp or f"/tmp/_blendkit_{os.getpid()}.exr"  # builders may run side by side
    sc.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(tmp, check_existing=False)
    w, h = im.size
    a = np.array(im.pixels[:], np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(im)
    np.save(path, a.astype(np.float16))
    return a


def camera(loc, target, lens=50.0, dof=None):
    import bpy
    from mathutils import Vector
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    cam.data.lens = lens
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    if dof:  # (focus distance m, f-stop)
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance, cam.data.dof.aperture_fstop = dof
    bpy.context.scene.camera = cam
    return cam


def light(kind, loc, energy, color=(1, 1, 1), size=0.1, target=None):
    import bpy
    from mathutils import Vector
    bpy.ops.object.light_add(type=kind, location=loc)
    L = bpy.context.object
    L.data.energy, L.data.color = energy, color
    if kind in ("POINT", "SPOT"):
        L.data.shadow_soft_size = size
    elif kind == "AREA":
        L.data.size = size
    elif kind == "SUN":
        L.data.angle = size
    if target is not None:
        L.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return L


class Nodes:
    """Terse node building: n = Nodes(material); x = n.math('MULTIPLY', a, 2.0); n.link(x, socket)."""

    def __init__(self, tree):
        self.t = tree
        self.N, self.L = tree.nodes, tree.links

    def node(self, kind, props=None, **inputs):
        nd = self.N.new(kind)
        for k, v in (props or {}).items():  # e.g. noise_dimensions: set before the inputs it shows
            setattr(nd, k, v)
        for k, v in inputs.items():
            socks = [i for i in nd.inputs if i.name == k] or [i for i in nd.inputs if i.name == {"Fac": "Factor", "Factor": "Fac"}.get(k)]
            on = [i for i in socks if i.enabled]
            self.set((on or socks)[0], v)
        return nd

    def set(self, sock, v):
        if hasattr(v, "is_output") or hasattr(v, "links"):
            self.L.new(v, sock)
        elif hasattr(v, "outputs"):
            self.L.new(v.outputs[0], sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=0.0, clamp=False):
        nd = self.N.new("ShaderNodeMath")
        nd.operation = op
        nd.use_clamp = clamp
        self.set(nd.inputs[0], a)
        self.set(nd.inputs[1], b)
        return nd.outputs[0]

    def vmath(self, op, a, b=(0, 0, 0)):
        nd = self.N.new("ShaderNodeVectorMath")
        nd.operation = op
        self.set(nd.inputs[0], a)
        self.set(nd.inputs[1], b)
        return nd.outputs["Value"] if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else nd.outputs[0]

    def xyz(self, v):
        nd = self.N.new("ShaderNodeSeparateXYZ")
        self.set(nd.inputs[0], v)
        return nd.outputs

    def ramp(self, fac, stops):
        nd = self.N.new("ShaderNodeValToRGB")
        cr = nd.color_ramp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for e, (p, c) in zip(cr.elements, stops):
            e.position, e.color = p, (*c, 1) if len(c) == 3 else c
        self.set(nd.inputs[0], fac)
        return nd.outputs[0]


def material(name, base=(0.5, 0.5, 0.5), rough=0.5, metallic=0.0):
    import bpy
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m, Nodes(m.node_tree), b


def save_meta(out, **kw):
    json.dump(kw, open(f"{out}/meta.json", "w"), indent=1)


def look_at_euler(loc, target):
    from mathutils import Vector
    return (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


def deg(x):
    return math.radians(x)
