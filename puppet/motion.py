"""Motion helpers shared by all puppets (no drawing code here).

  Spring          critically damped follower (smooth, lagging arms, mic, fret slides)
  HeadMotion      performer-like head: holds poses for seconds, eases to new ones, accent nods,
                  irregular blinks; deterministic in t
  decay_since     envelope that jumps on an event and decays (drum hits, strums)

Lessons (see docs/TOOLKIT.md): beat-locked bobbing and constant noise read as mechanical; holds +
slow eased moves + a few accents read as alive. Keep overshoot small (zeta >= 0.85).
"""
import math

import numpy as np

FPS = 30


class Spring:
    """Critically damped follower: smooth, slightly lagging motion toward a moving target."""

    def __init__(self, x, omega=9.0, zeta=1.0):
        self.x, self.v, self.w, self.z = np.array(x, float), np.zeros_like(np.array(x, float)), omega, zeta

    def step(self, target, dt=1 / FPS):
        # semi-implicit Euler, substepped so stiff springs (w*dt large) stay stable
        n = max(1, int(math.ceil(self.w * dt / 0.25)))
        h = dt / n
        tg = np.asarray(target, float)
        for _ in range(n):
            a = self.w * self.w * (tg - self.x) - 2 * self.z * self.w * self.v
            self.v = self.v + a * h
            self.x = self.x + self.v * h
        return self.x

    def at(self, target, t):
        """Advance to time t (catching up frame by frame from the last call, with this target), so a
        frame looks the same whether it is rendered in a video or alone as a still. A jump back in
        time (a new shot that started earlier) or a long gap settles on the target."""
        last = getattr(self, "t", None)
        n = 1 if last is None else int(round((t - last) * FPS))
        if last is None or n < 0 or n > 4 * FPS:
            self.x, self.v = np.array(target, float), np.zeros_like(np.array(target, float))
        else:
            for _ in range(max(n, 0)):
                self.step(target)
        self.t = t
        return self.x


class HeadMotion:
    """Head and body motion like a performer, not a metronome: the head holds a pose, then moves
    to a new one (chosen per phrase) with a springy ease and a little overshoot; nods come as
    accents on chosen hits (fast down, slow up) instead of a constant bob; blinks at irregular
    times and when the head starts a big move. Deterministic in t (simulated from 0, cached)."""

    def __init__(self, seed, accents, duration, beat, turn_amp=0.7, tilt_amp=4.0, nod_px=10.0, hold=(2.5, 5.5)):
        rng = np.random.default_rng(seed)
        n = int(duration * FPS) + 2
        dt = 1 / FPS
        # pose targets: piecewise constant, changes on beats
        tgt = np.zeros((n, 2))
        t, cur = 0.0, np.zeros(2)
        changes = []
        while t < duration:
            cur = np.array([np.clip(rng.normal(0, 0.55), -1, 1) * turn_amp, np.clip(rng.normal(0, 0.6), -1, 1) * tilt_amp])
            d = rng.uniform(*hold)
            t1 = round((t + d) / beat) * beat
            tgt[int(t * FPS):int(t1 * FPS) + 1] = cur
            changes.append(t)
            t = max(t1, t + beat)
        # springy follow (slightly underdamped) + a tiny drift
        x, v = np.zeros(2), np.zeros(2)
        w, z = 4.2, 0.88  # slow, barely any overshoot
        self.pose = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        for i in range(n):
            a = w * w * (tgt[i] - x) - 2 * z * w * v
            v = v + a * dt
            x = x + v * dt
            self.pose[i], self.vel[i] = x, v
        tt = np.arange(n) / FPS
        self.pose[:, 0] += 0.04 * turn_amp * np.sin(tt * 1.1 + seed)
        self.pose[:, 1] += 0.15 * tilt_amp * np.sin(tt * 0.7 + 2 * seed)
        # accent nods: fast attack (~60 ms), slow release (~400 ms), with velocity
        nod = np.zeros(n)
        for (ta, amp) in accents:
            i0 = int(ta * FPS)
            for k in range(0, int(0.9 * FPS)):
                if i0 + k < n:
                    u = k / FPS
                    nod[i0 + k] += amp * (1 - math.exp(-u / 0.06)) * math.exp(-u / 0.35)
        self.nod = np.clip(nod, 0, 1.4) * nod_px
        # blinks: irregular, plus at the start of big head moves
        bl = []
        t = rng.uniform(0.5, 2.0)
        while t < duration:
            bl.append(t)
            t += rng.uniform(2.0, 5.5)
        for c in changes:
            if rng.uniform() < 0.6:
                bl.append(c + 0.03)
        self.blinks = np.array(sorted(bl))

    def at(self, t):
        i = min(int(t * FPS), len(self.pose) - 1)
        turn, tilt = self.pose[i]
        j = np.searchsorted(self.blinks, t) - 1
        blink = 0.0
        if j >= 0:
            u = t - self.blinks[j]
            if u < 0.16:
                blink = min(1.0, u / 0.05) if u < 0.05 else max(0.0, 1 - (u - 0.05) / 0.11)
        sway = -self.vel[i, 0]  # hair lags behind the head's turn
        return float(turn), float(tilt), float(self.nod[i]), blink, float(sway)


def decay_since(t, times, tau):
    """exp(-(t - last event)/tau) for the most recent event before t (0 if none)."""
    i = np.searchsorted(times, t, side="right") - 1
    return math.exp(-(t - times[i]) / tau) if i >= 0 else 0.0, (t - times[i]) if i >= 0 else 9.0




class Path:
    """A position over time from keyframes [(t, x), ...]: holds between keys, eased moves (no jumps)."""

    def __init__(self, keys, ramp=None):
        """ramp=None: smoothstep over each move. ramp=seconds: walking profile, i.e. speed up over
        `ramp`, cruise at a constant speed, slow down over `ramp` (real walks cruise)."""
        self.keys, self.ramp = sorted(keys), ramp

    def x(self, t):
        k = self.keys
        if t <= k[0][0]:
            return float(k[0][1])
        for (t0, x0), (t1, x1) in zip(k[:-1], k[1:]):
            if t <= t1:
                T, tau = max(1e-6, t1 - t0), t - t0
                if self.ramp is None:
                    u = tau / T
                    e = u * u * (3 - 2 * u)
                else:
                    r = min(self.ramp, T / 2)
                    vm = 1.0 / (T - r)  # fraction of the move per second while cruising
                    if tau < r:
                        e = 0.5 * vm * tau * tau / r
                    elif tau > T - r:
                        e = 1.0 - 0.5 * vm * (T - tau) ** 2 / r
                    else:
                        e = 0.5 * vm * r + vm * (tau - r)
                return float(x0 + (x1 - x0) * e)
        return float(k[-1][1])

    def v(self, t, dt=1 / 60):
        return (self.x(t + dt) - self.x(t - dt)) / (2 * dt)


class Walk:
    """Front-view walk along a Path (the cut-out style: the body faces us, the legs step).
    Each foot stays planted while the body passes over it, then swings to the next spot with a
    lift; the hips bob twice per cycle. All in screen pixels for one character instance.

      walk = Walk(path, stride_px, lift_px)
      walk.feet(t) -> {"l": (foot_x, lift), "r": ...}, bob_px, speed (0..1)"""

    def __init__(self, path, stride, lift, bob=None, v_full=None, stance=0.5, leg=None):
        """stance: fraction of the cycle a foot is on the floor (0.5 cut-out style; ~0.6 real walk,
        which gives double support). leg: leg length in px (for walk3q)."""
        self.path, self.stride, self.lift, self.stance, self.leg = path, stride, lift, stance, leg
        self.bob = bob if bob is not None else 0.12 * lift
        self.v_full = v_full or 1.6 * stride  # px/s that counts as a full walk

    def feet(self, t):
        x = self.path.x(t)
        sp = min(1.0, abs(self.path.v(t)) / self.v_full)
        d = x  # distance along the (horizontal) path
        st = self.stance
        out = {}
        for side, off in (("l", 0.0), ("r", 0.5)):
            u = d / self.stride + off
            n = math.floor(u)
            fr = u - n
            plant = lambda m: (m + st / 2 - off) * self.stride  # noqa: E731  foot spot: under the body mid-stance
            if fr < st:  # stance: foot on the floor
                fx, lift = plant(n), 0.0
            else:  # swing to the next spot: heel up behind first, then reach forward
                w = (fr - st) / (1 - st)
                e = w * w * (3 - 2 * w)
                fx = plant(n) + (plant(n + 1) - plant(n)) * e
                lift = 0.5 * (1 - math.cos(2 * math.pi * w)) * (1 - 0.25 * w)  # eases in and out (no pop)
            # standing still: feet settle under the hips
            out[side] = (x + (fx - x) * sp, lift * sp)
        bob = self.bob * sp * abs(math.sin(2 * math.pi * d / self.stride))
        return out, bob, sp


# ---------------------------------------------------------------- fluid motion (overlap, weight, drag)
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class AngleSpring(Spring):
    """A Spring for an angle (radians): always takes the short way round."""

    def at(self, target, t):
        cur = float(self.x) if getattr(self, "t", None) is not None else float(target)
        return super().at(cur + wrap(float(target) - cur), t)


class ArmFollow:
    """Overlapping action for a two-bone limb: the IK answer gives the *wanted* shoulder angle and
    elbow bend; the shoulder follows it with a spring, the elbow bend with a slower, bouncier one,
    so the forearm trails, overshoots a little and settles (follow-through). `stiff` (0..1) makes
    both springs fast, for when the hand must be exactly somewhere (a mic at the mouth)."""

    def __init__(self, bend=1.0, w_sh=9.0, z_sh=0.75, w_el=6.0, z_el=0.5):
        self.bend = bend
        self.sh = AngleSpring(0.0, w_sh, z_sh)
        self.el = AngleSpring(0.0, w_el, z_el)
        self.base = (w_sh, w_el)
        self.prev = None

    def at(self, S, target, L1, L2, t, stiff=0.0):
        from limb import ik
        E, Wr = ik(S, target, L1, L2, self.bend)
        a1 = math.atan2(E[1] - S[1], E[0] - S[0])
        rel = wrap(math.atan2(Wr[1] - E[1], Wr[0] - E[0]) - a1)
        k = 1 + 3 * stiff
        self.sh.w, self.el.w = self.base[0] * k, self.base[1] * k
        a1s = float(self.sh.at(a1, t))
        rels = float(self.el.at(rel, t))
        E = np.asarray(S, float) + L1 * np.array([math.cos(a1s), math.sin(a1s)])
        a2 = a1s + rels
        Wr = E + L2 * np.array([math.cos(a2), math.sin(a2)])
        om = 0.0
        if self.prev is not None and t > self.prev[0]:
            om = wrap(a2 - self.prev[1]) / (t - self.prev[0])
        self.prev = (t, a2)
        return E, Wr, a2, om


class Drag:
    """Hand (or any end part) drag: an angle that trails the parent's rotation and swings back,
    e.g. the wrist when the forearm swings. Returns radians to add to the child's angle."""

    def __init__(self, k=0.08, limit=0.55, w=11.0, z=0.45):
        self.k, self.limit = k, limit
        self.s = Spring(0.0, w, z)

    def at(self, omega, t):
        return float(self.s.at(max(-self.limit, min(self.limit, -self.k * omega)), t))


def gait_body(walk, t):
    """Weight shift for a front-view walk (same phase as walk.feet): the hips shift toward the
    stance foot and drop on the swing side, the shoulders counter-tilt, the body is lowest just
    after each foot contact and highest at the passing position. All scaled by walking speed.
    Returns dict(sway_px, bob_px (+ = down), hip_tilt, shoulder_tilt (radians), speed)."""
    x = walk.path.x(t)
    sp = min(1.0, abs(walk.path.v(t)) / walk.v_full)
    u = x / walk.stride
    leg = walk.leg or walk.lift / 0.16  # front walk: built with lift = 0.16 x leg length
    sway = -0.03 * leg * math.sin(2 * math.pi * u) * sp
    bob = 0.028 * leg * math.cos(4 * math.pi * (u - 0.06)) * sp
    hip = math.radians(2.0) * math.sin(2 * math.pi * u) * sp
    return dict(sway_px=sway, bob_px=bob, hip_tilt=hip, shoulder_tilt=-0.6 * hip, speed=sp)


def walk3q(walk, t, hips=None):
    """3/4 walk support: how far the hips drop. Lowest just after each contact (double support),
    highest at passing. The depth is what the legs need at double support (both feet half a
    stance apart), but the curve itself is a smooth cosine: computing it from the planted feet
    directly made the body jolt when a foot lifted off. Returns drop_px (>= 0)."""
    sp = min(1.0, abs(walk.path.v(t)) / walk.v_full)
    reach = 0.985 * walk.leg
    dx = 0.5 * walk.stance * walk.stride
    depth = reach - math.sqrt(max(reach * reach - dx * dx, 0.0))
    u = walk.path.x(t) / walk.stride
    return depth * sp * 0.5 * (1 + math.cos(4 * math.pi * (u - 0.05)))


def knee3q(hip, foot, a, b, toward, depth=0.5):
    """Knee for a 3/4 view: the knee bends forward, which is partly toward the camera, so only
    `depth` of its sideways offset shows; the bones get foreshortened to match.
    Returns E, up_k, fore_k (lengths relative to a and b)."""
    hip, foot = np.asarray(hip, float), np.asarray(foot, float)
    Ek = knee(hip, foot, a, b, toward)
    v = foot - hip
    u = v / max(np.hypot(*v), 1e-6)
    base = hip + u * float(np.dot(Ek - hip, u))
    E = base + depth * (Ek - base)
    return E, float(np.hypot(*(E - hip)) / a), float(np.hypot(*(foot - E)) / b)


def soft_reach(hip, foot, L, start=0.94):
    """Soft IK: the foot target is pulled in smoothly near full extension so a leg never snaps
    straight (the knee position is very sensitive there)."""
    v = np.asarray(foot, float) - hip
    d = float(np.hypot(*v))
    da = start * L
    if d <= da:
        return np.asarray(foot, float)
    ds = da + (L - da) * (1 - math.exp(-(d - da) / (L - da)))
    return hip + v * (ds / d)


def knee(hip, foot, a, b, toward):
    """Two-bone IK in the screen plane: knee for thigh a and shin b, bent toward `toward` (+1 = +x)."""
    v = np.asarray(foot, float) - hip
    d = min(max(np.hypot(*v), 1e-3), 0.999 * (a + b))
    ca = (a * a + d * d - b * b) / (2 * a * d)
    A = math.acos(max(-1.0, min(1.0, ca)))
    u = v / max(np.hypot(*v), 1e-6)
    for sgn in (1, -1):
        c, s_ = math.cos(sgn * A), math.sin(sgn * A)
        k = hip + a * np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]])
        if (k[0] - (hip[0] + foot[0]) / 2) * toward >= 0:
            return k
    return k


class Lean:
    """Leans into a move and settles past upright when the move stops (underdamped)."""

    def __init__(self, k_deg_per_px_s=0.012, max_deg=6.0, w=5.0, z=0.45):
        self.k, self.max = k_deg_per_px_s, max_deg
        self.s = Spring(0.0, w, z)

    def at(self, velocity, t):
        tgt = max(-self.max, min(self.max, self.k * velocity))
        return math.radians(float(self.s.at(tgt, t)))


class Face:
    """Expressions for a band member who doesn't sing (drummer, guitarist), driven by the song:
    a smile that grows in the chorus, brows up on accents (crash, accented strum), a squint of
    effort on strong hits, a mouth that hangs a little open on hits or mouths the words with the
    singer, and now and then a playful look (brows + smile). All smoothed; returns a dict for
    Puppet.head_frame (smile, brow, mouth) plus squint (add to blink).

      face = Face(song, accents=[t...], hits=[t...], singalong=0.0, seed=1)"""

    def __init__(self, song, accents=(), hits=(), singalong=0.0, base_smile=0.15, seed=1):
        self.song = song
        self.acc, self.hits = np.sort(np.asarray(accents, float)), np.sort(np.asarray(hits, float))
        self.singalong, self.base = singalong, base_smile
        self.chorus = [(c["t0"], c["t1"]) for c in song.data.get("sections", []) if c["name"] == "chorus"]
        rng = np.random.default_rng(seed)
        self.looks = np.cumsum(rng.uniform(4.0, 8.0, 40))
        self.sm = Spring(base_smile, 2.5, 1.0)
        self.env = song.vocal_drivers()[0] if singalong else None

    def at(self, t):
        ch = any(a - 0.3 <= t < b for a, b in self.chorus)
        look, _ = decay_since(t, self.looks, 0.9)
        smile = float(self.sm.at(self.base + (0.55 if ch else 0.0) + 0.4 * look, t))
        acc, _ = decay_since(t, self.acc, 0.45)
        hit, _ = decay_since(t, self.hits, 0.12)
        brow = min(1.0, 0.8 * acc + 0.6 * look)
        squint = 0.28 * hit * (1.0 if ch else 0.5)
        mouth = 0.18 * acc + 0.08 * hit
        if self.env is not None and ch:
            f = min(int(t * 30), len(self.env) - 1)
            mouth = max(mouth, self.singalong * float(self.env[f]))
        return dict(smile=smile, brow=brow, mouth=min(0.6, mouth), squint=squint)
