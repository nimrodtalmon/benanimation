"""Walking legs for any character: two leg drawings (hip, knee, ankle marked) on a 3/4 walk.

  legs = Legs(parts_dir, parts_json["legs"], width, scale, path)   # path: motion.Path of the body x
  legs.body(t)        -> dict(drop, sway, hip_tilt, shoulder_tilt, lean_v, speed, dir)  (body offsets)
  legs.draw(frame, t, hips={"l": (x, y), "r": ...}, floor_y)   # hips: where the body put them now
  legs.contacts(t0, t1) -> [(t, side)]   # foot contacts (for footstep sounds)

What makes it read as walking (lessons, see docs/TOOLKIT.md): planted feet stay put in the world,
knees bend toward the travel direction, 60% stance (double support), the bob is a smooth cosine
(lowest just after contact) with the depth the legs need, soft IK so a leg never snaps straight,
and the Path cruises at a constant speed (Path(..., ramp=0.5)).
"""
import math

import numpy as np

import warp
from limb import Limb
from motion import Walk, gait_body, knee3q, soft_reach, walk3q
from warp import load_rgba

W, H = warp.W, warp.H


class Legs:
    def __init__(self, parts_dir, legs_json, width, scale, path, stride_k=0.9, lift_k=0.07, stance=0.6,
                 spread=0.9, round_top=True):
        self.s, self.path, self.spread = scale, path, spread
        self.limbs, self.hip_body = {}, {}
        for side in ("l", "r"):
            lj = legs_json[side]
            img = load_rgba(f"{parts_dir}/{lj['file']}")
            if round_top:  # round the top of the thigh about the hip (no corners poking out)
                img = img.copy()
                hx, hy = int(lj["hip"][0]), int(lj["hip"][1])
                if hy > 0:
                    yy, xx = np.mgrid[0:hy, 0:img.shape[1]]
                    keep = np.clip((0.56 * width - np.hypot(xx - hx, yy - hy)) / 12.0 + 0.5, 0, 1)
                    img[:hy, :, 3] = (img[:hy, :, 3] * keep).astype(img.dtype)
            jj = dict(shoulder=lj["hip"], elbow=lj["knee"], wrist=lj["ankle"])
            self.limbs[side] = Limb(img, img, jj, width, inner=1 if side == "l" else -1, fold_shadow=0.0)
            self.hip_body[side] = lj["hip_body"]
        L = self.limbs["l"]
        self.leg_px = (L.L1 + L.L2) * scale
        self.walk = Walk(path, stride=stride_k * self.leg_px, lift=lift_k * self.leg_px, stance=stance,
                         leg=self.leg_px, v_full=0.6 * self.leg_px)
        self.dir = 1.0

    def body(self, t):
        gb = gait_body(self.walk, t)
        v = self.path.v(t)
        if abs(v) > 5:
            self.dir = float(np.sign(v))
        gb["bob_px"] = walk3q(self.walk, t)
        gb["dir"] = self.dir
        gb["v"] = v
        return gb

    def draw(self, frame, t, hips, floor_y, cx, order=("l", "r")):
        """hips: screen hip points (after the body pose); floor_y: screen y of the soles at rest;
        cx: the body's center x (the walk path) -- each foot steps under its own hip."""
        feet, _, _ = self.walk.feet(t)
        s = self.s
        for side in order:
            L = self.limbs[side]
            hip = np.asarray(hips[side], float)
            fx, lift = feet[side]
            off = self.spread * (hip[0] - cx)
            leg = (L.L1 + L.L2) * s
            foot = np.array([fx + off, floor_y - lift * self.walk.lift])
            foot = soft_reach(hip, foot, leg, start=0.97)
            E, L.up_k, L.fore_k = knee3q(hip, foot, L.L1 * s, L.L2 * s, self.dir)
            lay, (x, y) = L.render(hip, E, foot, s, (W, H))
            if lay is not None:
                warp.over(frame, lay, x, y)
        return feet

    def contacts(self, t0, t1, fps=30):
        return contacts(self.walk, t0, t1, fps)


def contacts(walk, t0, t1, fps=30):
    """Foot contacts [(t, side)] of a motion.Walk between t0 and t1 (for footstep sounds)."""
    out, prev = [], None
    for f in range(int(t0 * fps), int(t1 * fps)):
        feet, _, sp = walk.feet(f / fps)
        up = {k: v[1] > 0.02 for k, v in feet.items()}
        if prev is not None:
            for k in up:
                if prev[k] and not up[k] and sp > 0.2:
                    out.append((f / fps, k))
        prev = up
    return out
