"""A rigged hand with real fingers (stand-in until Ben draws palm and finger parts).

The hand is a small 3D skeleton (palm + 4 fingers x 3 joints + thumb x 3), projected
orthographically and drawn back to front as tapered capsules with an ink outline and soft
painted shading. Held objects (a drumstick, a mic handle, a pick) live in the same 3D frame, so
fingers wrap around them and the depth order comes out right (thumb in front, palm behind).

Units: hand length (wrist to the middle fingertip, straight) = 1. Right hand; x across (thumb at
+x when the back of the hand faces the viewer and the fingers point down), y along the fingers
(+y = toward the fingertips), z toward the viewer (back of the hand faces +z at rest).
"""
import math

import cv2
import numpy as np

# finger bases (MCP knuckles), segment lengths, radii
FINGERS = dict(
    index=dict(base=(0.140, 0.450, 0.0), seg=(0.215, 0.125, 0.095), r=(0.046, 0.041, 0.036, 0.031), spread=-0.06),
    middle=dict(base=(0.047, 0.465, 0.0), seg=(0.235, 0.145, 0.100), r=(0.048, 0.043, 0.038, 0.032), spread=-0.01),
    ring=dict(base=(-0.048, 0.455, 0.0), seg=(0.220, 0.135, 0.098), r=(0.045, 0.040, 0.035, 0.030), spread=0.04),
    pinky=dict(base=(-0.135, 0.425, 0.0), seg=(0.170, 0.100, 0.088), r=(0.040, 0.035, 0.031, 0.027), spread=0.10),
)
THUMB = dict(base=(0.165, 0.11, -0.035), seg=(0.20, 0.155, 0.125), r=(0.062, 0.050, 0.044, 0.037))
PALM = [(0.19, 0.03), (-0.17, 0.03), (-0.19, 0.42), (-0.13, 0.47), (0.18, 0.48), (0.215, 0.40)]


def _rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def orient(yaw=0.0, pitch=0.0, roll=0.0):
    """Hand rotation: roll about the hand's long axis (y), pitch about x (tilts fingers toward or
    away from the viewer), yaw about z (in-plane turn). Angles in degrees."""
    return _rot((0, 0, 1), math.radians(yaw)) @ _rot((1, 0, 0), math.radians(pitch)) @ _rot((0, 1, 0), math.radians(roll))


def chain(base, d, n, segs, flex):
    """Joint positions of a finger: start at base, direction d, dorsal normal n; each joint bends
    by flex[k] toward the palm side (away from n)."""
    pts = [np.array(base, float)]
    d, n = np.array(d, float), np.array(n, float)
    for L, th in zip(segs, flex):
        c, s = math.cos(th), math.sin(th)
        d, n = d * c - n * s, n * c + d * s
        pts.append(pts[-1] + L * d)
    return pts, n


class Pose:
    """Finger curls: dict name -> (mcp, pip, dip) in degrees; thumb -> (cmc, mcp, ip); spread."""

    def __init__(self, curls, thumb=(20, 20, 15), thumb_dir=(0.55, 0.75, -0.35), spread=1.0, thumb_hidden=False):
        self.curls, self.thumb, self.thumb_dir, self.spread = curls, thumb, thumb_dir, spread
        self.thumb_hidden = thumb_hidden  # e.g. behind a guitar neck

    @staticmethod
    def mix(a, b, t):
        cu = {k: tuple(x * (1 - t) + y * t for x, y in zip(a.curls[k], b.curls[k])) for k in a.curls}
        th = tuple(x * (1 - t) + y * t for x, y in zip(a.thumb, b.thumb))
        td = tuple(x * (1 - t) + y * t for x, y in zip(a.thumb_dir, b.thumb_dir))
        return Pose(cu, th, td, a.spread * (1 - t) + b.spread * t)


def curl(c, mcp=78, pip=100, dip=62):
    return (c * mcp, c * pip, c * dip)


POSES = dict(
    relaxed=Pose({k: curl(0.25) for k in FINGERS}, thumb=(10, 15, 10)),
    fist=Pose({k: curl(1.0) for k in FINGERS}, thumb=(45, 40, 20), thumb_dir=(0.65, 0.76, 0.0), spread=0.3),
    # drumstick, matched grip: thumb and index pinch the stick (fulcrum), back fingers wrap loosely
    stick=Pose(dict(index=curl(0.75, 70, 95, 50), middle=curl(0.95), ring=curl(1.0), pinky=curl(1.0)),
               thumb=(45, 10, 5), thumb_dir=(0.15, 0.95, -0.25), spread=0.4),
    stick_loose=Pose(dict(index=curl(0.65, 70, 95, 50), middle=curl(0.70), ring=curl(0.62), pinky=curl(0.55)),
                     thumb=(42, 12, 8), thumb_dir=(0.15, 0.95, -0.25), spread=0.7),
    mic=Pose({k: curl(1.0, 82, 102, 66) for k in FINGERS}, thumb=(45, 40, 20), thumb_dir=(0.65, 0.76, 0.0), spread=0.2),
    fret=Pose(dict(index=(55, 80, 35), middle=(50, 85, 40), ring=(48, 85, 40), pinky=(40, 75, 35)),
              thumb=(15, 10, 5), thumb_dir=(0.45, 0.85, -0.3), spread=1.3),
    pick=Pose(dict(index=curl(0.70, 70, 80, 40), middle=curl(0.80), ring=curl(0.85), pinky=curl(0.85)),
              thumb=(40, 15, 10), thumb_dir=(0.10, 0.95, -0.25), spread=0.5),
)


class Style:
    def __init__(self, skin, ink=(0.10, 0.08, 0.08), ink_w=0.022, shade=0.35, nail=None, flat=False, light=(-0.45, -0.55, 0.70)):
        self.skin = np.array(skin, np.float32)  # BGR 0..1
        self.ink = np.array(ink, np.float32)
        self.ink_w, self.shade, self.flat = ink_w, shade, flat
        self.nail = np.array(nail if nail is not None else np.clip(self.skin * 1.08 + 0.05, 0, 1), np.float32)
        L = np.array(light, float)
        self.L = L / np.linalg.norm(L)


class Hand:
    """Pose a hand, then render() returns a BGRA layer (with the held object) and its offset.
    Every part is a chain of spheres in a z-buffer: overlaps resolve per pixel, shading comes from
    the sphere normals, and ink lines are drawn where the depth jumps (silhouettes and where one
    finger passes in front of another), not where two parts of one finger meet."""

    def __init__(self, style, mirror=False):
        self.st, self.mirror = style, mirror

    def parts(self, pose):
        """3D parts in hand space: dict(id, p0, p1, r0, r1, zs (depth flattening), color, nail_n)."""
        out = []
        # palm: a rounded slab (two wide flattened capsules across the hand)
        out.append(dict(id="palm", grp="palm", p0=(-0.085, 0.13, 0.0), p1=(-0.075, 0.37, 0.0), r0=0.11, r1=0.105, zs=0.32))
        out.append(dict(id="palm", grp="palm", p0=(0.085, 0.13, 0.0), p1=(0.095, 0.38, 0.0), r0=0.11, r1=0.11, zs=0.32))
        for name, f in FINGERS.items():
            sp = f["spread"] * pose.spread
            d = np.array([math.sin(-sp), math.cos(sp), 0.0])
            pts, n = chain(f["base"], d, (0, 0, 1), f["seg"], [math.radians(a) for a in pose.curls[name]])
            for k in range(3):
                out.append(dict(id=f"{name}{k}", grp=name, seg=k, p0=pts[k], p1=pts[k + 1], r0=f["r"][k] * 1.12, r1=f["r"][k + 1] * 1.12, zs=0.85,
                                nail_n=n if k == 2 else None))
        td = np.array(pose.thumb_dir, float)
        td /= np.linalg.norm(td)
        bend = np.array([-1.0, 0.15, -0.55])
        bend -= td * (bend @ td)
        nt = -bend / np.linalg.norm(bend)
        pts, n = chain(THUMB["base"], td, nt, THUMB["seg"], [math.radians(a) for a in pose.thumb])
        for k in range(3):
            if not getattr(pose, "thumb_hidden", False) or k == 0:
                out.append(dict(id=f"thumb{k}", grp="thumb", seg=k, p0=pts[k], p1=pts[k + 1], r0=THUMB["r"][k], r1=THUMB["r"][k + 1], zs=0.8,
                                nail_n=n if k == 2 else None))
        # the thumb's muscle at the base joins it to the palm
        out.append(dict(id="palm", grp="palm", p0=(0.06, 0.10, -0.01), p1=pts[1] * 0.55 + pts[0] * 0.45, r0=0.08, r1=0.058, zs=0.55))
        return out

    def render(self, pose, R, size, at, anchor=(0.0, 0.0, 0.0), held=(), ss=3):
        """R: 3x3 hand orientation (orient()); size: hand length in px; at: world position of the
        hand-space point `anchor`. held: parts in hand space: dict(p0, p1, r0, r1, color, gloss,
        mesh) - capsules (p1 = p0 for a ball)."""
        st = self.st
        Mr = np.diag([-1.0, 1, 1]) if self.mirror else np.eye(3)
        T = R @ Mr
        P = lambda p: T @ np.asarray(p, float)  # noqa: E731
        parts = self.parts(pose)
        for k, h in enumerate(held):
            q = dict(h)
            q.setdefault("id", f"held{k}")
            q.setdefault("zs", 1.0)
            parts.append(q)
        caps = []
        for pi, q in enumerate(parts):
            caps.append((P(q["p0"]), P(q["p1"]), q["r0"], q["r1"], q.get("zs", 1.0), pi))
        ext = np.array([c[0][:2] for c in caps] + [c[1][:2] for c in caps])
        rad = np.array([max(c[2], c[3]) for c in caps] * 2)[:, None]
        lo = (ext - rad).min(0) - 0.05
        hi = (ext + rad).max(0) + 0.05
        k = size * ss
        w, h = int((hi[0] - lo[0]) * k) + 2, int((hi[1] - lo[1]) * k) + 2
        zbuf = np.full((h, w), -1e9, np.float32)
        ibuf = np.full((h, w), -1, np.int32)
        nbuf = np.zeros((h, w, 3), np.float32)
        tbuf = np.zeros((h, w), np.float32)
        for a, b, r0, r1, zs, pi in caps:
            A = np.array([(a[0] - lo[0]) * k, (a[1] - lo[1]) * k, a[2] * k])
            B = np.array([(b[0] - lo[0]) * k, (b[1] - lo[1]) * k, b[2] * k])
            ra, rb = r0 * k, r1 * k
            x0 = max(0, int(min(A[0] - ra, B[0] - rb)) - 1)
            x1 = min(w, int(max(A[0] + ra, B[0] + rb)) + 2)
            y0 = max(0, int(min(A[1] - ra, B[1] - rb)) - 1)
            y1 = min(h, int(max(A[1] + ra, B[1] + rb)) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            gx, gy = gx + 0.5, gy + 0.5
            v = B[:2] - A[:2]
            vv = float(v @ v)
            t = np.clip(((gx - A[0]) * v[0] + (gy - A[1]) * v[1]) / vv, 0, 1) if vv > 1e-6 else np.zeros_like(gx)
            cx, cy, cz = A[0] + v[0] * t, A[1] + v[1] * t, A[2] + (B[2] - A[2]) * t
            r = ra + (rb - ra) * t
            dx, dy = gx - cx, gy - cy
            d2 = r * r - dx * dx - dy * dy
            inside = d2 > 0
            dz = np.sqrt(np.maximum(d2, 0))
            z = cz + zs * dz
            zb = zbuf[y0:y1, x0:x1]
            win = inside & (z > zb)
            if not win.any():
                continue
            zb[win] = z[win]
            ibuf[y0:y1, x0:x1][win] = int(pi)
            tbuf[y0:y1, x0:x1][win] = t[win]
            nz = dz / np.maximum(r, 1e-3)
            nn = np.stack([dx / np.maximum(r, 1e-3), dy / np.maximum(r, 1e-3), nz / max(zs, 0.2) * 0.6 + nz * 0.4], -1)
            nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-6
            nbuf[y0:y1, x0:x1][win] = nn[win]
        hidden_ids = [i for i, q in enumerate(parts) if q.get("hidden")]
        cover = (ibuf >= 0) & ~np.isin(ibuf, hidden_ids)  # hidden parts (e.g. a guitar neck) occlude but are not drawn
        # the palm's surface: smooth its depth and take normals from it (no creases between parts)
        palm_ids = [i for i, q in enumerate(parts) if q.get("grp") == "palm"]
        pm = np.isin(ibuf, palm_ids)
        if pm.any():
            sg = 0.045 * k
            zf = np.where(pm, zbuf, 0).astype(np.float32)
            num = cv2.GaussianBlur(zf, (0, 0), sg)
            den = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), sg) + 1e-6
            zs_ = num / den
            gx_ = cv2.Sobel(zs_, cv2.CV_32F, 1, 0, ksize=3) / 8
            gy_ = cv2.Sobel(zs_, cv2.CV_32F, 0, 1, ksize=3) / 8
            nn = np.stack([-gx_, -gy_, np.ones_like(gx_)], -1)
            nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
            # blend toward the analytic normal at the silhouette so the rim still turns away
            rim = np.clip(1 - cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 3) / (0.05 * k), 0, 1)[..., None]
            nbuf = np.where(pm[..., None], nn * (1 - rim) + nbuf * rim, nbuf)
        # color per pixel
        base = np.zeros((h, w, 3), np.float32)
        gloss = np.zeros((h, w), np.float32)
        nailm = np.zeros((h, w), np.float32)
        for pi, q in enumerate(parts):
            m = ibuf == pi
            if not m.any():
                continue
            base[m] = q.get("color", st.skin)
            gloss[m] = q.get("gloss", 0.08)
            nn = q.get("nail_n")
            if nn is not None:
                nd = P(nn)  # dorsal direction of the fingertip (view space)
                nd = nd / (np.linalg.norm(nd) + 1e-9)
                al = nbuf[..., 0] * nd[0] + nbuf[..., 1] * nd[1] + nbuf[..., 2] * nd[2]
                nailm[m & (tbuf > 0.35) & (al > 0.72)] = 1.0
            if q.get("mesh"):
                gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
                per = q["r0"] * k * 0.22
                mesh = ((np.abs(((gx + gy) / per) % 1 - 0.5) < 0.12) | (np.abs(((gx - gy) / per) % 1 - 0.5) < 0.12)).astype(np.float32)
                base[m] *= (1 - 0.35 * mesh[m])[:, None]
        Lv = st.L
        diff = np.clip(nbuf @ Lv.astype(np.float32), 0, 1)
        if st.flat:
            tone = np.where(diff > 0.35, 1.0, 0.82).astype(np.float32)
            col = base * tone[..., None]
        else:
            amb = 1 - st.shade
            col = base * (amb + st.shade * diff)[..., None]
            # warm, slightly red shadows on skin (light through the skin)
            col = col + (1 - diff)[..., None] * np.array([0.0, 0.0, 0.05], np.float32) * (gloss < 0.2)[..., None]
            Hh = Lv + np.array([0, 0, 1.0])
            Hh = Hh / np.linalg.norm(Hh)
            spec = np.clip(nbuf @ Hh.astype(np.float32), 0, 1) ** 24 * gloss
            col = col + spec[..., None]
        # contact shadows: darker where a nearer part covers the neighborhood (between fingers, under
        # the thumb, where the hand grips the object) - this is what makes the parts touch
        zb = np.where(cover, zbuf, -1e9).astype(np.float32)
        r_ao = max(3, int(0.05 * k)) | 1
        zmax = cv2.dilate(zb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_ao, r_ao)))
        ao = np.clip((zmax - zb) / (0.09 * k), 0, 1) * cover
        ao = cv2.GaussianBlur(ao.astype(np.float32), (0, 0), 0.02 * k)
        col = col * (1 - (0.22 if st.flat else 0.38) * ao[..., None])
        # skin grain: soft blotches and a fine marker/paint grain, so it is not a smooth 3D surface
        if not hasattr(self, "_grain") or self._grain.shape != (h, w):
            rng = np.random.default_rng(7)
            g1 = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.05 * k)
            g2 = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=0.012 * k)
            self._grain = (g1 / (g1.std() + 1e-6)) * 0.045 + (g2 / (g2.std() + 1e-6)) * 0.012
        skin_px = (gloss < 0.2)[..., None]
        col = col * (1 + self._grain[..., None] * skin_px)
        col = col + skin_px * np.clip(1 - diff, 0, 1)[..., None] * np.array([0.0, 0.01, 0.04], np.float32)  # warm in the shade
        nm = cv2.GaussianBlur(nailm, (0, 0), 0.8)[..., None]
        col = col * (1 - 0.8 * nm) + st.nail * 0.8 * nm
        # ink: silhouette, any big depth jump, and where two different fingers meet
        grp = np.array([hash(q.get("grp", q["id"])) % 100003 for q in parts] + [-1])
        segk = np.array([q.get("seg", -1) for q in parts] + [-9])
        is_palm = np.array([q.get("grp") == "palm" for q in parts] + [False])
        gb = grp[ibuf]
        sk = segk[ibuf]
        pb = is_palm[ibuf]
        big, small = 0.05 * k, 0.006 * k
        edge = np.zeros((h, w), bool)
        for sh in ((0, 1), (1, 0), (1, 1), (1, -1)):
            zb2 = np.roll(zbuf, sh, (0, 1))
            gb2, sk2, pb2 = np.roll(gb, sh, (0, 1)), np.roll(sk, sh, (0, 1)), np.roll(pb, sh, (0, 1))
            ib2 = np.roll(ibuf, sh, (0, 1))
            dzz = np.abs(zb2 - zbuf)
            joined = (pb & (sk2 == 0)) | (pb2 & (sk == 0))  # knuckles flow into the back of the hand
            edge |= (dzz > big) | ((gb != gb2) & ~joined & (ib2 >= 0) & (ibuf >= 0))
        nail_edge = (cv2.morphologyEx((nailm > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
        ink = max(1, int(round(st.ink_w * k)))
        em = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ink, ink))).astype(np.float32)
        em = np.maximum(em, nail_edge.astype(np.float32) * 0.6)
        alpha = cv2.dilate(cover.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ink, ink))).astype(np.float32)
        if hidden_ids:  # no ink halo spilling onto the hidden object
            hid = np.isin(ibuf, hidden_ids)
            alpha[hid & ~cover] = 0
            em[hid] = 0
        img = np.zeros((h, w, 4), np.float32)
        img[..., :3] = np.clip(col, 0, 1) * (1 - em[..., None]) + st.ink * em[..., None]
        img[..., :3] = np.where(cover[..., None], img[..., :3], st.ink)
        img[..., 3] = alpha
        img = cv2.resize(img, (max(1, w // ss), max(1, h // ss)), interpolation=cv2.INTER_AREA)
        img[..., :3] = np.where(img[..., 3:4] > 1e-3, img[..., :3] / np.maximum(img[..., 3:4], 1e-3) * np.minimum(img[..., 3:4], 1) , 0)
        a3 = P(anchor)
        a2 = np.array([(a3[0] - lo[0]) * k, (a3[1] - lo[1]) * k]) / ss
        off = np.asarray(at, float) - a2
        self.last = dict(lo=lo, k=k / ss, P=P, off=off)
        return img, (int(math.floor(off[0])), int(math.floor(off[1]))), off - np.floor(off)

    def world(self, p):
        """World position of a hand-space point, after the last render()."""
        L = self.last
        q = L["P"](p)
        return np.array([(q[0] - L["lo"][0]) * L["k"], (q[1] - L["lo"][1]) * L["k"]]) + L["off"]
