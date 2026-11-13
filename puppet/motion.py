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


