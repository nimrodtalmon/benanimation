"""Generic production: stage, crowd, camera finish (push, vignette, grain, credits), and a
shot-list renderer that renders a clip in parallel slices and muxes the song.

A clip script defines: a Song, a cast (name -> factory), and SHOTS [(t0, t1, shot)] where a shot
is a cast name or a tuple of names drawn together; then calls Production(...).main(sys.argv).
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np

import warp

ROOT = warp.ROOT
FPS, W, H = warp.FPS, warp.W, warp.H


def stage_light(frame, t):
    """Darken toward the floor (the light comes from above) and add a little haze."""
    if not hasattr(stage_light, "ramp"):
        yy = np.linspace(0, 1, H, dtype=np.float32)
        stage_light.ramp = (1 - 0.7 * np.clip((yy - 0.72) / 0.28, 0, 1) ** 1.3)[:, None, None]
    frame *= stage_light.ramp
    return frame


def finish(frame, t, f, n, rng, credit, zoom=(1.0, 1.06), focus=(W / 2, H * 0.42)):
    z = zoom[0] + (zoom[1] - zoom[0]) * f / max(1, n - 1)
    C = cv2.getRotationMatrix2D((focus[0] + 8 * math.sin(t * 0.8), focus[1]), 0, z)
    C[0, 2] += W / 2 - focus[0]  # the focus point is brought to the frame center (a pan)
    C[1, 2] += H * 0.42 - focus[1]
    frame = cv2.warpAffine(frame, C, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if not hasattr(finish, "vign"):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        finish.vign = (1 - 0.45 * np.clip(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2, 0, 1))[..., None]
    frame = frame * finish.vign + rng.normal(0, 0.018, (H, W, 1)).astype(np.float32)
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    for i, line in enumerate(credit):
        cv2.putText(frame, line, (30, H - 24 - 22 * (len(credit) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def writer(out, start, dur, audio):
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(start), "-t", str(dur), "-i", audio,
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out], stdin=subprocess.PIPE)




class Stage:
    """Dark stage: back wall gradient, colored light beams from above (one pulses per bar),
    soft bokeh, a floor. Precomputed once; per frame only intensities change."""

    COLORS = [(0.95, 0.45, 0.35), (0.35, 0.55, 1.0), (0.40, 0.95, 0.70), (0.95, 0.35, 0.85)]  # BGR

    def __init__(self, song):
        self.song = song
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        self.base = np.broadcast_to((0.07 + 0.09 * yy) * np.array([1.25, 0.9, 1.0], np.float32), (H, W, 3)).copy()
        self.base[int(H * 0.82):] *= 0.55  # floor edge
        hw, hh = W // 4, H // 4
        self.beams = []
        for i, (x, ang) in enumerate([(0.12, 18), (0.38, 6), (0.62, -6), (0.88, -18)]):
            m = np.zeros((hh, hw), np.float32)
            top = np.array([x * hw, -5])
            L = hh * 1.3
            a = math.radians(90 + ang)
            spread = math.radians(9)
            p1 = top + L * np.array([math.cos(a - spread), math.sin(a - spread)])
            p2 = top + L * np.array([math.cos(a + spread), math.sin(a + spread)])
            cv2.fillConvexPoly(m, np.round(np.array([top + [-2, 0], top + [2, 0], p2, p1])).astype(np.int32), 1.0, cv2.LINE_AA)
            fall = np.clip(1.0 - np.linspace(0, 1, hh, dtype=np.float32) * 0.6, 0, 1)[:, None]
            m = cv2.GaussianBlur(m * fall, (0, 0), 6)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            self.beams.append((m, np.array(self.COLORS[i], np.float32)))

    def frame(self, t, wide=False, cue=None):
        """cue (optional): dict(beams=0..1 colored lights, spot=(x, k) a white spotlight at x,
        house=0..1 overall level); a clip sets it per frame (e.g. lights coming up on the chorus)."""
        cue = cue or {}
        SPB, BAR = self.song.spb, self.song.bar
        ph = (t % SPB) / SPB
        beat = math.exp(-ph * 5)
        bar = int(t / BAR) % 4
        img = self.base.copy()
        bk = cue.get("beams", 1.0)
        for i, (m, col) in enumerate(self.beams):
            k = 0.10 + (0.30 * beat if i == bar else 0.04 * math.sin(t * 0.9 + i))
            img += m * col * k * 1.3 * bk
        if cue.get("spot"):
            x, k = cue["spot"]
            img += self.spot(x) * k
        return img * cue.get("house", 1.0)

    def spot(self, x):
        """A white spotlight from above at screen x: soft cone and a pool on the floor."""
        if not hasattr(self, "_spot"):
            hw, hh = W // 4, H // 4
            m = np.zeros((hh, hw * 3), np.float32)
            c = hw * 1.5
            cv2.fillConvexPoly(m, np.round(np.array([[c - 6, -5], [c + 6, -5], [c + hw * 0.16, hh * 0.86], [c - hw * 0.16, hh * 0.86]])).astype(np.int32), 0.22, cv2.LINE_AA)
            cv2.ellipse(m, (int(c), int(hh * 0.88)), (int(hw * 0.2), int(hh * 0.05)), 0, 0, 360, 0.55, -1, cv2.LINE_AA)
            m = cv2.GaussianBlur(m, (0, 0), 5)
            self._spot = cv2.resize(m, (W * 3, H), interpolation=cv2.INTER_LINEAR)[..., None] * np.array([0.95, 0.97, 1.0], np.float32)
        x0 = int(1.5 * W - x)
        x0 = max(0, min(2 * W, x0))
        return self._spot[:, x0:x0 + W]


def crowd(frame, t, spb, n=13, h=240, seed=4):
    """Out-of-focus crowd at the bottom (heads and shoulders, some hands up), bobbing on the beat."""
    rng = np.random.default_rng(seed)
    ph = (t % spb) / spb
    hs = 4  # draw at quarter size, blur, scale up: soft foreground silhouettes
    m = np.zeros((H // hs, W // hs), np.float32)
    for i in range(n):
        x = (i + rng.uniform(-0.3, 0.3)) * W / (n - 1)
        r = rng.uniform(0.8, 1.2) * h * 0.30
        lag = rng.uniform(0, 0.2)
        bob = 14 * math.exp(-(((ph - lag) % 1) * 5))
        y = H - h * rng.uniform(0.40, 0.62) + bob
        cv2.ellipse(m, (int(x / hs), int((y + r * 2.2) / hs)), (int(r * 1.9 / hs), int(r * 1.4 / hs)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.ellipse(m, (int(x / hs), int(y / hs)), (int(r * 0.85 / hs), int(r / hs)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        if rng.uniform() < 0.35:
            up = 40 * math.exp(-(((ph - lag) % 1) * 4))
            side = rng.choice([-1, 1])
            hx = x + side * r * 1.1
            tip = (hx + side * rng.uniform(5, 30), y - r * 1.8 - up)
            cv2.line(m, (int(hx / hs), int((y + r) / hs)), (int(tip[0] / hs), int(tip[1] / hs)), 1.0, max(1, int(r * 0.45 / hs)), cv2.LINE_AA)
            cv2.circle(m, (int(tip[0] / hs), int(tip[1] / hs)), max(1, int(r * 0.32 / hs)), 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 2.2)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    frame *= 1 - 0.96 * m
    frame += m * np.array([0.020, 0.012, 0.018], np.float32)



class HandDrawn:
    """A hand-drawn finish (look options): line boil (the whole image wobbles a pixel or two and
    re-draws every few frames, like traced drawings), paper grain multiplied in, and a color grade
    (lifted teal shadows, warm highlights, a gentle S-curve). Background areas are smooth, so the
    boil only shows on lines and texture."""

    def __init__(self, boil=0.4, hold=4, paper=0.07, grade=0.6, seed=0):  # boil 1.3 read as too wobbly
        self.boil, self.hold, self.paper, self.grade = boil, hold, paper, grade
        rng = np.random.default_rng(seed)
        g = rng.normal(0, 1, (H, W)).astype(np.float32)
        fib = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=0.8)
        blot = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 3)
        blot = cv2.resize(blot, (W, H))
        tex = 0.45 * g / g.std() + 0.35 * fib / fib.std() + 0.5 * blot / blot.std()
        self.tex = (tex / tex.std())[..., None]
        self._k, self._maps = None, None

    def maps(self, f):
        k = f // self.hold
        if k != self._k:
            rng = np.random.default_rng(1000 + k)
            hs = 8
            dx = cv2.GaussianBlur(rng.normal(0, 1, (H // hs, W // hs)).astype(np.float32), (0, 0), 2.2)
            dy = cv2.GaussianBlur(rng.normal(0, 1, (H // hs, W // hs)).astype(np.float32), (0, 0), 2.2)
            dx = cv2.resize(dx / (dx.std() + 1e-6), (W, H)) * self.boil
            dy = cv2.resize(dy / (dy.std() + 1e-6), (W, H)) * self.boil
            X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            self._maps, self._k = (X + dx, Y + dy), k
        return self._maps

    def __call__(self, frame, f):
        if self.boil > 0:
            mx, my = self.maps(f)
            frame = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if self.paper > 0:
            frame = frame * (1 + self.paper * self.tex * (0.35 + frame.mean(2, keepdims=True)))
        if self.grade > 0:
            lum = frame.mean(2, keepdims=True)
            sh = np.clip(1 - lum * 2.5, 0, 1) * np.array([0.035, 0.02, -0.01], np.float32)  # teal shadows (BGR)
            hi = np.clip(lum * 2 - 1, 0, 1) * np.array([-0.03, 0.0, 0.04], np.float32)  # warm highlights
            c = np.clip(frame, 0, 1)
            curve = c * c * (3 - 2 * c)
            frame = frame + self.grade * (0.35 * (curve - c) + sh + hi)
        return frame


class Production:
    def __init__(self, song, cast, shots, name, crowd_in=(), zoom_wide=(1.04, 1.0), zoom=(1.0, 1.05), script=None, cue=None,
                 look=None, twos=False, crowd_wide=True):
        """look: a HandDrawn finish (or None); twos: characters move on twos (15 drawings a second,
        the camera and lights stay smooth), the cadence of hand-drawn 2D."""
        self.song, self.cast_f, self.shots, self.name = song, cast, shots, name
        self.crowd_in = set(crowd_in)
        self.zoom_wide, self.zoom = zoom_wide, zoom
        self.script = script
        self.cue = cue  # cue(t, shot) -> Stage cue dict (lights per moment and shot)
        self.look, self.twos, self.crowd_wide = look, twos, crowd_wide
        self.cast = {}
        self.stage = Stage(song)

    def who(self, shot):
        names = shot if isinstance(shot, (tuple, list)) else (shot,)
        out = []
        for n in names:
            if n not in self.cast:
                self.cast[n] = self.cast_f[n]()
            out.append(self.cast[n])
        return out

    def shot_at(self, t):
        sh = next(sh for sh in self.shots if sh[0] <= t < sh[1])
        return sh if len(sh) == 4 else (sh[0], sh[1], sh[2], {})

    def frame(self, t, f, rng):
        """A shot is (t0, t1, who) or (t0, t1, who, opts); opts: zoom=(start, end), focus=(x, y) or
        focus(t) -> (x, y) for a pan."""
        a, b, shot, opts = self.shot_at(t)
        wide = isinstance(shot, (tuple, list))
        frame = self.stage.frame(t, wide, self.cue(t, shot) if self.cue else None)
        who = self.who(shot)
        tc, fc = (math.floor(t * 15) / 15, (f // 2) * 2) if self.twos else (t, f)
        for w in who:
            if hasattr(w, "advance"):
                w.advance(fc)
            w.draw(frame, tc)
        if (wide and self.crowd_wide) or shot in self.crowd_in:
            crowd(frame, t, self.song.spb)
        frame = stage_light(frame, t)
        if self.look is not None:
            frame = self.look(frame, f)
        k = (t - a) / (b - a)
        credit = list(dict.fromkeys([w.j["credit"] for w in who])) + [self.song.credit]
        zoom = opts.get("zoom", self.zoom_wide if wide else self.zoom)
        focus = opts.get("focus", (W / 2, H * 0.42))
        if callable(focus):
            focus = focus(t)
        return finish(frame, t, int(k * 100), 101, rng, credit, zoom=zoom, focus=focus)

    def render(self, t0, t1, out, frames=None):
        n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        rng = np.random.default_rng(n0)
        ff = None
        if frames is None:
            ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        for f in range(n0, n1):
            img = self.frame(f / FPS, f, rng)
            if ff:
                ff.stdin.write(img.tobytes())
            elif f in frames:
                cv2.imwrite(frames[f], img)
        if ff:
            ff.stdin.close()
            ff.wait()

    def stills(self, times, path):
        """Render single frames (for checking) into path_<t>.jpg."""
        for t in times:
            f = int(round(t * FPS))
            self.render(f / FPS, (f + 1) / FPS, None, frames={f: f"{path}_{t}.jpg"})

    def render_all(self, parts=2):
        total = self.song.duration
        cuts = [total * i / parts for i in range(parts + 1)]
        os.makedirs(f"{ROOT}/out", exist_ok=True)
        segs = [f"{ROOT}/out/{self.name}-seg{i}.mp4" for i in range(parts)]
        procs = [subprocess.Popen([sys.executable, self.script, "segment", str(cuts[i]), str(cuts[i + 1]), segs[i]]) for i in range(parts)]
        for p in procs:
            p.wait()
        lst = f"{ROOT}/out/{self.name}-segs.txt"
        open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
        out = f"{ROOT}/out/{self.name}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", self.song.audio,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
        print(out)
        return out

    def render_range(self, t0, t1, out):
        """A short piece with its audio (for tests and options)."""
        tmp = out + ".v.mp4"
        self.render(t0, t1, tmp)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ss", str(t0), "-t", str(t1 - t0), "-i", self.song.audio,
                        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", out], check=True)
        os.remove(tmp)
        return out

    def main(self, argv):
        what = argv[1] if len(argv) > 1 else "video"
        if what == "segment":  # segment t0 t1 [out] -> with audio
            t0, t1 = float(argv[2]), float(argv[3])
            out = argv[4] if len(argv) > 4 else f"{ROOT}/out/{self.name}-{argv[2]}-{argv[3]}.mp4"
            print(self.render_range(t0, t1, out))
        elif what == "stills":
            self.stills([float(a) for a in argv[2:]], f"{ROOT}/out/{self.name}-still")
        elif what == "video":
            self.render_all()
        elif what in self.cast_f:  # one character alone: <name> [start dur]
            start, dur = (float(argv[2]), float(argv[3])) if len(argv) > 3 else (8.0, 6.0)
            self.shots = [(0, self.song.duration + 1, what, {})]
            out = f"{ROOT}/out/{self.name}-{what}.mp4"
            tmp = out + ".v.mp4"
            self.render(start, start + dur, tmp)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ss", str(start), "-t", str(dur), "-i", self.song.audio,
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
            os.remove(tmp)
            print(out)
