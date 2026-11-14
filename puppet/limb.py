"""Bending limbs from drawings: an upper arm and a forearm (Ben's two pages, or one straight
arm cut at the elbow) bend like a sleeve, without a visible hinge.

How the elbow is built (all by looking pixels up in the drawing):
  - the upper arm and the forearm move rigidly (rotation about shoulder / elbow, from 2-bone IK);
  - the outer side of the elbow is filled by sweeping the arm's cross-section at the elbow
    around the elbow point, so the outer contour becomes a round arc made of the drawing's
    own outline and colors;
  - the forearm is drawn on top; on the inner side it keeps its round cap (if the drawing has
    one), which reads as the crease of the bent arm; a soft shadow darkens the fold.

  limb = Limb(upper_rgba, fore_rgba, joints, width, inner=+1)   # inner: side the elbow bends to
  E, W = limb.solve(S, target, scale)
  layer, (x0, y0) = limb.render(S, E, W, scale)
"""
import math

import cv2
import numpy as np


def ik(S, T, L1, L2, bend=1.0):
    """Two-bone IK: shoulder S, wrist target T, bone lengths -> (elbow, wrist). bend picks the side."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    d = T - S
    D0 = float(np.hypot(*d))
    D = min(max(D0, abs(L1 - L2) + 1e-3), L1 + L2 - 1e-3)
    a = math.atan2(d[1], d[0])
    c = (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D)
    b = math.acos(max(-1.0, min(1.0, c)))
    E = S + L1 * np.array([math.cos(a + bend * b), math.sin(a + bend * b)])
    u = (T - E) / max(1e-6, np.hypot(*(T - E)))
    return E, E + L2 * u


def _rot(theta, p0, p1, k, axis0=None, along=1.0):
    """Affine: rotate by theta about p0, scale by k, move p0 to p1. With axis0 (rest angle of a
    bone) and along < 1, the part is also squashed along its own axis (foreshortening)."""
    c, s = math.cos(theta) * k, math.sin(theta) * k
    R = np.array([[c, -s], [s, c]])
    if axis0 is not None and along != 1.0:
        u = np.array([math.cos(axis0), math.sin(axis0)])
        R = R @ (np.eye(2) + (along - 1.0) * np.outer(u, u))
    return np.hstack([R, (np.asarray(p1, float) - R @ np.asarray(p0, float))[:, None]])


def join_parts(upper, forearm, up_joints, fo_joints):
    """Ben's two drawings -> two layers in one shared rest frame, the forearm rotated so that
    it continues the upper arm straight, elbow on elbow. Returns (up_layer, fo_layer, joints)."""
    S, E = np.float32(up_joints["shoulder"]), np.float32(up_joints["elbow"])
    E2, W2 = np.float32(fo_joints["elbow"]), np.float32(fo_joints["wrist"])
    a_up = math.atan2(*(E - S)[::-1])
    a_fo = math.atan2(*(W2 - E2)[::-1])
    L2 = float(np.hypot(*(W2 - E2)))
    h, w = upper.shape[:2]
    fh, fw = forearm.shape[:2]
    pad = int(L2 + max(fh, fw))
    H, W = h + 2 * pad, w + 2 * pad
    up = np.zeros((H, W, 4), np.float32)
    up[pad:pad + h, pad:pad + w] = upper
    M = cv2.getRotationMatrix2D((float(E2[0]), float(E2[1])), math.degrees(a_fo - a_up), 1.0)
    M[:, 2] += (E + pad) - E2
    fo = cv2.warpAffine(forearm, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    tf = lambda p: (M @ np.array([p[0], p[1], 1.0])).tolist()  # noqa: E731
    joints = dict(shoulder=(S + pad).tolist(), elbow=(E + pad).tolist(), wrist=tf(W2))
    for k in ("grip", "tip"):
        if k in fo_joints:
            joints[k] = tf(fo_joints[k])
    a = np.maximum(up[..., 3], fo[..., 3])
    ys, xs = np.nonzero(a > 0.01)
    x0, y0, x1, y1 = xs.min() - 4, ys.min() - 4, xs.max() + 5, ys.max() + 5
    joints = {k: [v[0] - x0, v[1] - y0] for k, v in joints.items()}
    return up[y0:y1, x0:x1].copy(), fo[y0:y1, x0:x1].copy(), joints


class Limb:
    def __init__(self, upper, fore, joints, width, inner=1, fold_shadow=0.28):
        """upper, fore: BGRA float layers in one rest frame (fore may equal upper for a single
        painted arm: each part is cut at the elbow line). joints: shoulder, elbow, wrist [, grip]."""
        j = joints
        self.S0, self.E0, self.W0 = (np.array(j[k], np.float64) for k in ("shoulder", "elbow", "wrist"))
        self.grip0 = np.array(j.get("grip", j["wrist"]), np.float64)
        self.L1 = float(np.hypot(*(self.E0 - self.S0)))
        self.L2 = float(np.hypot(*(self.W0 - self.E0)))
        self.a1_0 = math.atan2(*(self.E0 - self.S0)[::-1])
        self.a2_0 = math.atan2(*(self.W0 - self.E0)[::-1])
        self.width, self.inner, self.fold_shadow = width, inner, fold_shadow
        self.fore_k = 1.0  # forearm foreshortening (1 = as drawn; < 1 = pointing toward the camera)
        self.up_k = 1.0  # same for the upper part (a thigh when the knee comes toward the camera)
        h, w = upper.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        u1 = (self.E0 - self.S0) / self.L1
        n1 = np.array([-u1[1], u1[0]])  # bending by a positive angle turns the forearm toward n1
        self.n_in = n1 * inner
        u2 = (self.W0 - self.E0) / self.L2
        a1 = (X - self.E0[0]) * u1[0] + (Y - self.E0[1]) * u1[1]
        a2 = (X - self.E0[0]) * u2[0] + (Y - self.E0[1]) * u2[1]
        lat_in = (X - self.E0[0]) * self.n_in[0] + (Y - self.E0[1]) * self.n_in[1]
        # upper arm: everything before the elbow line
        self.up = upper.copy()
        self.up[..., 3] *= np.clip(0.5 - a1, 0, 1)
        # forearm: after the elbow line; on the inner side its own cap stays (crease)
        self.fo = fore.copy()
        near_e = np.hypot(X - self.E0[0], Y - self.E0[1]) < 0.55 * width
        self.fo[..., 3] *= np.maximum(np.clip(0.5 + a2, 0, 1), np.clip(lat_in / 4.0, 0, 1) * near_e)
        # for the sweep: the upper arm's outer half in polar form around the elbow
        # (r = distance from the elbow axis, k = distance above the elbow line)
        self.n_out, self.u1 = -self.n_in, u1
        self.upper_src = upper

    def solve(self, S, T, scale):
        """IK for this arm (it bends toward its inner side): shoulder, wrist target -> elbow, wrist."""
        return ik(S, T, self.L1 * scale * self.up_k, self.L2 * scale * self.fore_k, -self.inner)

    def solve_reach(self, S, T, scale, slack=0.97):
        """Like solve, but if the target is out of reach with the current foreshortening, the
        forearm (then the upper arm) is drawn less foreshortened, up to full length, so the hand
        gets there (a foreshortened arm that cannot reach would leave a stick or pick in the air).
        Changes fore_k / up_k for this frame; render with the same Limb."""
        d = float(np.hypot(*(np.asarray(T, float) - S))) / max(scale, 1e-9)
        if d > slack * (self.L1 * self.up_k + self.L2 * self.fore_k):
            self.fore_k = min(1.0, max(self.fore_k, (d / slack - self.L1 * self.up_k) / self.L2))
            if d > slack * (self.L1 * self.up_k + self.L2 * self.fore_k):
                self.up_k = min(1.0, max(self.up_k, (d / slack - self.L2 * self.fore_k) / self.L1))
        return self.solve(S, T, scale)

    def _poses(self, S, E, Wr, scale):
        a1 = math.atan2(*(np.asarray(E, float) - S)[::-1])
        a2 = math.atan2(*(np.asarray(Wr, float) - E)[::-1])
        return (_rot(a1 - self.a1_0, self.S0, S, scale, self.a1_0, self.up_k),
                _rot(a2 - self.a2_0, self.E0, E, scale, self.a2_0, self.fore_k), a1, a2)

    def grip(self, S, E, Wr, scale):
        """World position of the grip point and the forearm angle (for a prop in the hand)."""
        _, M2, a1, a2 = self._poses(S, E, Wr, scale)
        return M2 @ np.append(self.grip0, 1), a2

    def render(self, S, E, Wr, scale, frame_size=None, part="all"):
        """part: 'all', or 'upper' / 'fore' to put other layers (body, drums, guitar) in between."""
        M1, M2, a1, a2 = self._poses(S, E, Wr, scale)
        h, w = self.up.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float)
        pts = np.vstack([corners @ M1.T, corners @ M2.T])
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        if frame_size is not None:
            x0, y0 = max(x0, -4), max(y0, -4)
            x1, y1 = min(x1, frame_size[0] + 4), min(y1, frame_size[1] + 4)
        if x1 <= x0 or y1 <= y0:
            return None, (0, 0)
        ww, hh = int(x1 - x0), int(y1 - y0)
        T = lambda M: np.hstack([M[:, :2], (M[:, 2] - [x0, y0])[:, None]])  # noqa: E731
        out = cv2.warpAffine(self.up, T(M1), (ww, hh), flags=cv2.INTER_LINEAR, borderValue=0)
        F = cv2.warpAffine(self.fo, T(M2), (ww, hh), flags=cv2.INTER_LINEAR, borderValue=0)
        # the swept outer elbow: points around E between the two outer normals
        bend = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi  # >0: forearm turned toward +n1
        ex, ey = E[0] - x0, E[1] - y0
        r_px = int(self.width * 0.75 * scale) + 3
        bx0, by0 = int(max(0, ex - r_px)), int(max(0, ey - r_px))
        bx1, by1 = int(min(ww, ex + r_px)), int(min(hh, ey + r_px))
        if bx1 > bx0 and by1 > by0 and abs(bend) > 0.01:
            yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            vx, vy = xx - ex, yy - ey
            r = np.hypot(vx, vy) / scale
            phi = np.arctan2(vy, vx)
            o1 = math.atan2(-self.n_in[1], -self.n_in[0]) + (a1 - self.a1_0)  # upper arm's outer normal, world
            rel = (phi - o1 + math.pi) % (2 * math.pi) - math.pi
            t = rel / bend
            inside = ((t >= -0.03) & (t <= 1.03)).astype(np.float32)
            # sample the drawing at the same distance from the arm's axis (so the outline lands
            # on the arc) and a little above the elbow line (so the texture is not smeared)
            psi = 0.35 * np.sin(np.pi * np.clip(t, 0, 1))
            sx = (self.E0[0] + r * self.n_out[0] - (1.5 + r * psi) * self.u1[0]).astype(np.float32)
            sy = (self.E0[1] + r * self.n_out[1] - (1.5 + r * psi) * self.u1[1]).astype(np.float32)
            col = cv2.remap(self.upper_src, sx, sy, cv2.INTER_LINEAR, borderValue=0)
            col[..., 3] *= inside
            reg = out[by0:by1, bx0:bx1]
            a = col[..., 3:4]
            reg[..., :3] = reg[..., :3] * (1 - a) + col[..., :3] * a
            reg[..., 3:4] = np.maximum(reg[..., 3:4], a)
            if self.fold_shadow > 0:  # soft shadow in the fold, growing with the bend
                d_in = -(vx * math.cos(o1) + vy * math.sin(o1)) / (scale * self.width)
                k = self.fold_shadow * min(1.0, abs(bend) / 1.4) * np.exp(-(r / (0.55 * self.width)) ** 2) * np.clip(d_in * 3, 0, 1)
                reg[..., :3] *= (1 - k)[..., None]
        if part == "upper":
            return out, (int(x0), int(y0))
        if part == "fore":
            return F, (int(x0), int(y0))
        a = F[..., 3:4]
        out[..., :3] = out[..., :3] * (1 - a) + F[..., :3] * a
        out[..., 3:4] = out[..., 3:4] * (1 - a) + a
        return out, (int(x0), int(y0))
