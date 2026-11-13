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

    def frame(self, t, wide=False):
        SPB, BAR = self.song.spb, self.song.bar
        ph = (t % SPB) / SPB
        beat = math.exp(-ph * 5)
        bar = int(t / BAR) % 4
        img = self.base.copy()
        for i, (m, col) in enumerate(self.beams):
            k = 0.10 + (0.30 * beat if i == bar else 0.04 * math.sin(t * 0.9 + i))
            img += m * col * k * 1.3
        return img


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



class Production:
    def __init__(self, song, cast, shots, name, crowd_in=(), zoom_wide=(1.04, 1.0), zoom=(1.0, 1.05), script=None):
        self.song, self.cast_f, self.shots, self.name = song, cast, shots, name
        self.crowd_in = set(crowd_in)
        self.zoom_wide, self.zoom = zoom_wide, zoom
        self.script = script
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
        return next(sh for sh in self.shots if sh[0] <= t < sh[1])

    def frame(self, t, f, rng):
        a, b, shot = self.shot_at(t)
        wide = isinstance(shot, (tuple, list))
        frame = self.stage.frame(t, wide)
        who = self.who(shot)
        for w in who:
            if hasattr(w, "advance"):
                w.advance(f)
            w.draw(frame, t)
        if wide or shot in self.crowd_in:
            crowd(frame, t, self.song.spb)
        frame = stage_light(frame, t)
        k = (t - a) / (b - a)
        credit = list(dict.fromkeys([w.j["credit"] for w in who])) + [self.song.credit]
        return finish(frame, t, int(k * 100), 101, rng, credit, zoom=self.zoom_wide if wide else self.zoom)

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

    def main(self, argv):
        what = argv[1] if len(argv) > 1 else "video"
        if what == "segment":
            self.render(float(argv[2]), float(argv[3]), argv[4])
        elif what == "stills":
            self.stills([float(a) for a in argv[2:]], f"{ROOT}/out/{self.name}-still")
        elif what == "video":
            self.render_all()
        elif what in self.cast_f:  # one character alone: <name> [start dur]
            start, dur = (float(argv[2]), float(argv[3])) if len(argv) > 3 else (8.0, 6.0)
            self.shots = [(0, self.song.duration + 1, what)]
            out = f"{ROOT}/out/{self.name}-{what}.mp4"
            tmp = out + ".v.mp4"
            self.render(start, start + dur, tmp)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ss", str(start), "-t", str(dur), "-i", self.song.audio,
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
            os.remove(tmp)
            print(out)
