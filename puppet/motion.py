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
        a = self.w * self.w * (np.asarray(target, float) - self.x) - 2 * self.z * self.w * self.v
        self.v = self.v + a * dt
        self.x = self.x + self.v * dt
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

    def __init__(self, keys):
        self.keys = sorted(keys)

    def x(self, t):
        k = self.keys
        if t <= k[0][0]:
            return float(k[0][1])
        for (t0, x0), (t1, x1) in zip(k[:-1], k[1:]):
            if t <= t1:
                u = (t - t0) / max(1e-6, t1 - t0)
                e = u * u * (3 - 2 * u)
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

    def __init__(self, path, stride, lift, bob=None, v_full=None):
        self.path, self.stride, self.lift = path, stride, lift
        self.bob = bob if bob is not None else 0.12 * lift
        self.v_full = v_full or 1.6 * stride  # px/s that counts as a full walk

    def feet(self, t):
        x = self.path.x(t)
        sp = min(1.0, abs(self.path.v(t)) / self.v_full)
        d = x  # distance along the (horizontal) path
        out = {}
        for side, off in (("l", 0.0), ("r", 0.5)):
            u = d / self.stride + off
            n = math.floor(u)
            fr = u - n
            plant = lambda m: (m + 0.25 - off) * self.stride  # noqa: E731  foot spot (distance) for step m
            if fr < 0.5:  # stance: foot on the floor
                fx, lift = plant(n), 0.0
            else:  # swing to the next spot
                w = (fr - 0.5) / 0.5
                e = w * w * (3 - 2 * w)
                fx = plant(n) + (plant(n + 1) - plant(n)) * e
                lift = math.sin(math.pi * w)
            # standing still: feet settle under the hips
            out[side] = (x + (fx - x) * sp, lift * sp)
        bob = self.bob * sp * abs(math.sin(2 * math.pi * d / self.stride))
        return out, bob, sp
