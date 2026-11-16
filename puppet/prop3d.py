"""Hand props in 3D (Blender), rendered once side-on and drawn like everything 3D (outlines from the
geometry when a drawing style is on), then placed in 2D by their anchors:

  python3 puppet/prop3d.py build rifle      -> out/propcache/rifle[@style]-<hash>/
  pr = Prop3D("rifle")
  pr.draw(frame, at=(x, y), anchor="grip", angle=deg, scale=1.0, flip=False)
  pr.point("muzzle", at, "grip", angle, scale, flip)     -> screen xy of another anchor

From public/props/<name>.json (units cm, x along the prop, z up): parts are "profile" (a 2D outline
extruded to a width), "box", "cyl" (tapered), "ball", "ring"; materials by name; anchors (hands,
muzzle...). The render is orthographic from the side at px_per_cm, transparent.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blendkit as bk  # noqa: E402

BUILD = 2
ROOT = bk.ROOT


def spec_of(name):
    return json.load(open(f"{ROOT}/public/props/{name}.json"))


def cache_dir(name):
    st = bk.style()
    return bk.cache_dir("prop", name + (f"@{st}" if st else ""), spec_of(name), BUILD)


def bounds(S):
    xs, zs = [], []
    for p in S["parts"]:
        if p["kind"] == "profile":
            xs += [q[0] for q in p["points"]]
            zs += [q[1] for q in p["points"]]
        elif p["kind"] in ("box", "cyl"):
            r = max(p.get("r0", 0), p.get("r1", 0))
            for q in (p["from"], p["to"]):
                xs += [q[0] - r, q[0] + r]
                zs += [q[1] - r, q[1] + r]
        else:
            r = p["r"] + p.get("thick", 0)
            xs += [p["at"][0] - r, p["at"][0] + r]
            zs += [p["at"][1] - r, p["at"][1] + r]
    m = 3.0
    return min(xs) - m, max(xs) + m, min(zs) - m, max(zs) + m


# ---------------------------------------------------------------- Blender side
def _mat(name, M):
    m, n, b = bk.material(name, tuple(M["color"]), M.get("rough", 0.5), M.get("metallic", 0.0))
    if M.get("grain"):
        tc = n.node("ShaderNodeTexCoord").outputs["Object"]
        wave = n.node("ShaderNodeTexWave", Vector=n.vmath("MULTIPLY", tc, (0.08, 1.0, 1.0)), Scale=3.0, Distortion=6.0, Detail=3.0)
        c = np.array(M["color"])
        n.set(b.inputs["Base Color"], n.ramp(wave.outputs[1], [(0.2, tuple(c * 0.6)), (0.8, tuple(c * 1.25))]))
    return m


def build(name):
    import bpy  # noqa: I001  (bpy first: it provides bmesh)
    import bmesh
    S = spec_of(name)
    out = cache_dir(name)
    os.makedirs(out, exist_ok=True)
    x0, x1, z0, z1 = bounds(S)
    k = S["px_per_cm"]
    res = (int((x1 - x0) * k), int((z1 - z0) * k))
    sc = bk.scene(res, S.get("samples", 32), transparent=True)
    if bk.style():
        bk.drawn(sc, line_px=2.2, color=(0.06, 0.055, 0.055), crease=140.0, wobble=0.8)
    mats = {nm: _mat(nm, M) for nm, M in S["materials"].items()}
    cm = 0.01

    def obj(me, nm):
        o = bpy.data.objects.new(nm, me)
        sc.collection.objects.link(o)
        return o

    for p in S["parts"]:
        kd, mat = p["kind"], mats[p["mat"]]
        dy = p.get("dy", 0.0) * cm
        if kd == "profile":
            bm = bmesh.new()
            w = p["width"] * cm / 2
            vs = [bm.verts.new((q[0] * cm, -w, q[1] * cm)) for q in p["points"]]
            f = bm.faces.new(vs)
            r = bmesh.ops.extrude_face_region(bm, geom=[f])
            for v in [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]:
                v.co.y += 2 * w
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            me = bpy.data.meshes.new(p["name"])
            bm.to_mesh(me)
            bm.free()
            o = obj(me, p["name"])
            bev = o.modifiers.new("b", "BEVEL")
            bev.width, bev.segments = 0.4 * cm, 2
        elif kd == "box":
            a, b = np.array(p["from"]) * cm, np.array(p["to"]) * cm
            bpy.ops.mesh.primitive_cube_add(location=((a[0] + b[0]) / 2, dy, (a[1] + b[1]) / 2))
            o = bpy.context.object
            o.scale = (abs(b[0] - a[0]) / 2, p["width"] * cm / 2, abs(b[1] - a[1]) / 2)
            bev = o.modifiers.new("b", "BEVEL")
            bev.width, bev.segments = 0.25 * cm, 2
        elif kd == "cyl":
            a, b = np.array(p["from"]) * cm, np.array(p["to"]) * cm
            d = b - a
            L = float(np.hypot(*d))
            bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=p["r0"] * cm, radius2=p["r1"] * cm, depth=L,
                                            location=((a[0] + b[0]) / 2, dy, (a[1] + b[1]) / 2))
            o = bpy.context.object
            o.rotation_euler = (0, math.atan2(d[0], d[1]), 0)  # cone axis (z) -> along d in the xz plane
        elif kd == "ball":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=p["r"] * cm, location=(p["at"][0] * cm, dy, p["at"][1] * cm))
            o = bpy.context.object
        elif kd == "ring":
            bpy.ops.mesh.primitive_torus_add(major_radius=p["r"] * cm, minor_radius=p["thick"] * cm,
                                             location=(p["at"][0] * cm, dy, p["at"][1] * cm), rotation=(math.pi / 2, 0, 0))
            o = bpy.context.object
        else:
            raise ValueError(kd)
        o.data.materials.append(mat)
        if kd in ("cyl", "ball", "ring"):
            bpy.ops.object.shade_smooth()
    cx, cz = (x0 + x1) / 2 * cm, (z0 + z1) / 2 * cm
    cam = bk.camera((cx, -3.0, cz), (cx, 0, cz), 50)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = (x1 - x0) * cm
    bk.light("AREA", (cx - 0.3, -1.2, cz + 1.2), 60, (1.0, 0.96, 0.9), size=1.5, target=(cx, 0, cz))
    bk.light("AREA", (cx + 0.6, -1.0, cz - 0.4), 15, (0.85, 0.9, 1.0), size=1.5, target=(cx, 0, cz))
    bk.world(0.4, (0.5, 0.5, 0.5))
    bk.render(f"{out}/side.npy")
    px = lambda q: [(q[0] - x0) * k, (z1 - q[1]) * k]  # noqa: E731
    views = {"side": dict(size=list(res), anchors={a: px(q) for a, q in S["anchors"].items()})}
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    for vn, V in S.get("views", {}).items():  # perspective views (e.g. pointing at the camera)
        sc.render.resolution_x, sc.render.resolution_y = V["res"]
        cam.data.type = "PERSP"
        cam.data.lens = V["lens"]
        cam.location = Vector(V["eye"]) * cm
        cam.rotation_euler = bk.look_at_euler(Vector(V["eye"]) * cm, Vector(V["target"]) * cm)
        bpy.context.view_layer.update()
        corners = [Vector((x * cm, y * cm, z * cm)) for x in (x0, x1) for y in (-3, 3) for z in (z0, z1)]
        for _ in range(3):  # zoom so the prop fills `fit` of the picture
            cs = [world_to_camera_view(sc, cam, c) for c in corners]
            ext = max(max(c.x for c in cs) - min(c.x for c in cs), max(c.y for c in cs) - min(c.y for c in cs))
            cam.data.lens *= V.get("fit", 0.9) / max(ext, 1e-3)
            bpy.context.view_layer.update()
        bk.render(f"{out}/{vn}.npy")
        anc = {}
        for a, q in S["anchors"].items():
            c = world_to_camera_view(sc, cam, Vector((q[0] * cm, 0, q[1] * cm)))
            anc[a] = [c.x * V["res"][0], (1 - c.y) * V["res"][1]]
        views[vn] = dict(size=list(V["res"]), anchors=anc)
    bk.save_meta(out, name=name, size=list(res), anchors=views["side"]["anchors"], views=views, px_per_cm=k)
    print("prop built:", out)
    return out


# ---------------------------------------------------------------- 2D side
class Prop3D:
    def __init__(self, name):
        import ink
        d = bk.built(cache_dir(name))
        if d is None:
            import subprocess
            subprocess.run([sys.executable, os.path.abspath(__file__), "build", name], check=True)
            d = bk.built(cache_dir(name))
        self.meta = json.load(open(f"{d}/meta.json"))
        views = self.meta.get("views", {"side": dict(anchors=self.meta["anchors"])})
        self.imgs, self.anc = {}, {}
        e = None
        for vn, V in views.items():
            a = np.load(f"{d}/{vn}.npy").astype(np.float32)
            al = np.clip(a[..., 3:4], 0, 1)
            lin = a[..., :3] / np.maximum(al, 1e-3)
            if e is None:  # one exposure for all views (from the side view)
                e = 0.6 / max(1e-5, float(np.percentile(lin.max(-1)[al[..., 0] > 0.5], 90)))
            disp = bk.tonemap(lin, e)[..., ::-1]
            img = ink.flat_cached(f"{d}/{vn}.npy", disp, al[..., 0]) if ink.active() else np.concatenate([disp, al], 2)
            ys, xs = np.nonzero(img[..., 3] > 0.01)  # crop to the prop (+ a margin)
            y0, x0 = max(0, ys.min() - 8), max(0, xs.min() - 8)
            img = img[y0:ys.max() + 9, x0:xs.max() + 9]
            self.imgs[vn] = np.ascontiguousarray(img, np.float32)  # BGRA
            self.anc[vn] = {k: np.array(v, float) - [x0, y0] for k, v in V["anchors"].items()}
        self.img, self.anchors = self.imgs["side"], self.anc["side"]
        self.k = self.meta["px_per_cm"]

    def fit(self, view, a, pa, b, pb):
        """2x3 map for `view` putting anchor a on pa and anchor b on pb (rotation + uniform scale)."""
        A, B = self.anc[view][a], self.anc[view][b]
        d0, d1 = B - A, np.asarray(pb, float) - np.asarray(pa, float)
        sc = np.hypot(*d1) / max(1e-6, np.hypot(*d0))
        ang = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
        c, s = math.cos(ang) * sc, math.sin(ang) * sc
        Mm = np.array([[c, -s], [s, c]])
        return np.hstack([Mm, (np.asarray(pa, float) - Mm @ A)[:, None]])

    def draw_at(self, frame, view, anchor, xy, width, angle=0.0, tint=None):
        """Draw `view` with `anchor` on xy, `width` px wide (its cropped picture), turned by angle
        (deg, counterclockwise); returns {anchor: screen xy}."""
        import props
        img = self.imgs[view]
        sc = width / img.shape[1]
        a = math.radians(-angle)
        c, s = math.cos(a) * sc, math.sin(a) * sc
        Mm = np.array([[c, -s], [s, c]])
        M = np.hstack([Mm, (np.asarray(xy, float) - Mm @ self.anc[view][anchor])[:, None]])
        if tint is not None:
            img = np.concatenate([img[..., :3] * tint, img[..., 3:]], 2)
        props.blit(frame, img, M)
        return {k: M[:, :2] @ v + M[:, 2] for k, v in self.anc[view].items()}

    def draw_fit(self, frame, view, a, pa, b, pb, tint=None):
        """Draw `view` with anchor a on pa and b on pb; returns {anchor: screen xy}."""
        import props
        M = self.fit(view, a, pa, b, pb)
        img = self.imgs[view]
        if tint is not None:
            img = np.concatenate([img[..., :3] * tint, img[..., 3:]], 2)
        props.blit(frame, img, M)
        return {k: M[:, :2] @ v + M[:, 2] for k, v in self.anc[view].items()}

    def matrix(self, at, anchor, angle, scale, flip=False):
        """2x3 map from the render to the screen: `anchor` lands on `at`, rotated by angle (deg, the
        prop's x axis on screen, counterclockwise from screen right), scaled; flip mirrors it."""
        a = math.radians(-angle)
        c, s = math.cos(a) * scale, math.sin(a) * scale
        R = np.array([[c, -s], [s, c]])
        F = np.diag([-1.0, 1.0]) if flip else np.eye(2)
        Mm = R @ F
        p = self.anchors[anchor]
        return np.hstack([Mm, (np.asarray(at, float) - Mm @ p)[:, None]])

    def point(self, name, at, anchor, angle, scale, flip=False):
        M = self.matrix(at, anchor, angle, scale, flip)
        return M[:, :2] @ self.anchors[name] + M[:, 2]

    def draw(self, frame, at, anchor, angle, scale, flip=False, tint=None):
        import props
        img = self.img if tint is None else np.concatenate([self.img[..., :3] * tint, self.img[..., 3:]], 2)
        props.blit(frame, img, self.matrix(at, anchor, angle, scale, flip))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
