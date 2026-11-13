"""Warp puppets: animate a drawing by bending its own pixels (the Live2D / VTuber idea),
instead of moving stiff cut-out parts.

A character = a head layer and a body layer (RGBA, cut from one drawing) + a few
landmarks (face oval, mouth, eyes, chin, neck pivot). Per frame:
  - head turn: features inside the face oval slide sideways, most at the center (fake 3D)
  - mouth: the jaw region is pulled down; a painted mouth interior fills the gap
  - blink: real skin from just below the eye slides over it, lash line at its edge
  - hair sway: pixels outside the face oval and low on the head swing with a lag
  - tilt / nod / breathing: small rigid moves on top
Drivers come from the song (vocal loudness, mouth shapes, beat).

  python puppet/warp.py pepper   -> out/warp-pepper.mp4
  python puppet/warp.py distor   -> out/warp-distor.mp4
"""
import json
import math
import subprocess
import sys

import cv2
import numpy as np

FPS, W, H = 30, 1920, 1080
ROOT = "/home/claude/benanimation"

CHARS = {
    "pepper": dict(
        head="public/paint/pepper-head.png", body="public/paint/pepper-body.png", src_h=2207, height=1500, top=-30,
        face=(905, 850, 235, 265), pivot=(900, 1090), mouth=(910, 922, 42, 36), chin=995,
        eyes=[(790, 750, 60, 40), (1012, 748, 58, 40)], lid_from=118, lash=(42, 29, 31), lash_w=8,
        hair_from=820, turn_px=26, bg="pepper", credit='Painting: "Portrait of Pepper" by David Revoy, CC BY 4.0, animated'),
    "distor": dict(
        head="public/chars/distor-front-head.png", body="public/chars/distor-front-body.png", src_h=1600, height=1350, top=-60,
        face=(535, 735, 255, 300), pivot=(520, 1000), mouth=(562, 862, 105, 60), chin=1015,
        eyes=[(420, 556, 118, 58), (737, 578, 120, 74)], lid_from=150, lash=(20, 18, 19), lash_w=16,
        lid_color=(745, 735), hair_from=700, turn_px=18, bg="stage", credit="Drawing: Ben"),
}


def load(path, scale):
    im = cv2.imread(f"{ROOT}/{path}", cv2.IMREAD_UNCHANGED)
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0  # BGRA


# ---------- drivers from the song ----------
def drivers(start, n):
    song = json.load(open(f"{ROOT}/src/song.json"))
    sr = 16000
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "/home/claude/stems/vox_ex.wav", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True).stdout
    y = np.frombuffer(raw, np.float32)
    hop = sr // FPS
    rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(len(y) // hop)])
    rms /= np.percentile(rms, 95)
    env = np.zeros_like(rms)
    for i in range(len(rms)):  # fast attack, ~70 ms release
        prev = env[i - 1] if i else 0
        env[i] = rms[i] if rms[i] > prev else prev * 0.62 + rms[i] * 0.38
    env = np.clip((env - 0.12) / 0.88, 0, 1) ** 0.8
    vowels = []
    for f in range(n):
        t = start + f / FPS
        v = "ah"
        for c in song["vocal"]:
            if c["t"] <= t < c["t"] + c["dur"] + 0.05:
                v = c["vowel"]
        vowels.append(v)
    i0 = int(start * FPS)
    return env[i0:i0 + n], vowels


def smooth_noise(t, seed):
    return sum(math.sin(t * f + seed * p) * a for f, p, a in [(0.61, 1.3, 0.55), (1.37, 2.1, 0.3), (2.9, 0.7, 0.15)])


# ---------- the warp ----------
class Puppet:
    def __init__(self, cfg):
        self.c = cfg
        s = cfg["height"] / cfg["src_h"]
        self.s = s
        self.head = load(cfg["head"], s)
        self.body = load(cfg["body"], s)
        h, w = self.head.shape[:2]
        self.X, self.Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        S = lambda v: v * s  # noqa: E731
        self.face = [S(v) for v in cfg["face"]]
        self.pivot = [S(v) for v in cfg["pivot"]]
        self.mouth = [S(v) for v in cfg["mouth"]]
        self.chin = S(cfg["chin"])
        self.eyes = [[S(v) for v in e] for e in cfg["eyes"]]
        self.lid_from = S(cfg["lid_from"])
        self.hair_from = S(cfg["hair_from"])
        X, Y = self.X, self.Y
        fx, fy, frx, fry = self.face
        u, v = (X - fx) / frx, (Y - fy) / fry
        r2 = u * u + v * v
        self.w_turn = np.clip(1 - r2, 0, 1) ** 0.8
        self.w_out = np.clip(r2 - 1.0, 0, 1)
        mx, my, mw, J = self.mouth
        gx = np.exp(-((X - mx) / (mw * 1.5)) ** 2)
        below = np.clip((Y - my) / (mw * 0.3), 0, 1)
        fade = np.exp(-np.clip((Y - self.chin) / (mw * 0.9), 0, None) ** 2)
        self.w_jaw = gx * below * fade
        self.w_lip = gx * np.exp(-((my - Y) / (mw * 0.3)) ** 2) * (Y < my)
        self.w_hair = np.clip((Y - self.hair_from) / (fry * 0.9), 0, 1) * self.w_out

    def head_frame(self, turn, mouth, vowel, blink, sway):
        mx, my, mw, J = self.mouth
        width = {"ee": 1.15, "ah": 1.0, "oh": 0.75, "oo": 0.55}.get(vowel, 1.0)
        drop = J * mouth * (0.55 if vowel == "ee" else 1.0)
        dx = turn * self.c["turn_px"] * self.s * 2 * (self.w_turn - 0.25 * self.w_out) + sway * self.w_hair
        dy = drop * self.w_jaw - 0.12 * drop * self.w_lip
        dx = dx + (width - 1) * 0.0  # width handled by the interior shape
        img = cv2.remap(self.head, self.X - dx, self.Y - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mxt = mx + turn * self.c["turn_px"] * self.s * 2
        if drop > 1.0:
            self.mouth_interior(img, mxt, my, mw * width * (0.75 + 0.25 * mouth), drop)
        if blink > 0.02:
            self.blink(img, blink, turn)
        return img

    def mouth_interior(self, img, mx, my, half, drop):
        xs = np.linspace(-1, 1, 40)
        g = (1 - xs ** 2) ** 0.7
        up = my - 0.12 * drop * g - 1
        lo = my + drop * g * 0.92
        pts = np.concatenate([np.stack([mx + xs * half, up], 1), np.stack([mx + xs[::-1] * half, lo[::-1]], 1)]).astype(np.int32)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(mask, [pts], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.6)
        col = np.zeros((h, w, 3), np.float32)
        col[:] = (0.13, 0.10, 0.24)  # BGR dark red-brown
        teeth = np.zeros((h, w), np.float32)
        if drop > 0.35 * self.mouth[3]:
            tpts = np.array([[mx - half * 0.7, my - 2], [mx + half * 0.7, my - 2], [mx + half * 0.6, my + drop * 0.22], [mx - half * 0.6, my + drop * 0.22]], np.int32)
            cv2.fillPoly(teeth, [tpts], 1.0)
            teeth = cv2.GaussianBlur(teeth, (0, 0), 1.2)
        tongue = np.zeros((h, w), np.float32)
        cv2.ellipse(tongue, (int(mx), int(my + drop * 0.72)), (int(half * 0.55), max(1, int(drop * 0.22))), 0, 0, 360, 1.0, -1)
        tongue = cv2.GaussianBlur(tongue, (0, 0), 2)
        col = col * (1 - teeth[..., None]) + np.array([0.80, 0.86, 0.90], np.float32) * teeth[..., None]
        col = col * (1 - 0.8 * tongue[..., None]) + np.array([0.30, 0.30, 0.62], np.float32) * 0.8 * tongue[..., None]
        a = mask[..., None]
        img[..., :3] = img[..., :3] * (1 - a) + col * a

    def blink(self, img, b, turn):
        h, w = img.shape[:2]
        src = np.zeros_like(img)
        if "lid_color" in self.c:  # flat marker drawings: plain skin color, slightly textured
            cx, cy = (int(v * self.s) for v in self.c["lid_color"])
            col = np.median(self.head[cy - 8:cy + 8, cx - 8:cx + 8, :3].reshape(-1, 3), axis=0)
            if not hasattr(self, "lid_tex"):
                self.lid_tex = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 0.02, (h, w)).astype(np.float32), (0, 0), 1.5)[..., None]
            src[..., :3] = col + self.lid_tex
        else:
            off = int(self.lid_from)
            src[:-off] = img[off:]
        for ex, ey, erx, ery in self.eyes:
            ex = ex + turn * self.c["turn_px"] * self.s * 2 * 0.9
            m = np.zeros((h, w), np.float32)
            cv2.ellipse(m, (int(ex), int(ey - 4 * self.s)), (int(erx + 12 * self.s), int(ery + 16 * self.s)), 0, 0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 5 * self.s + 1)
            yline = ey - ery + 2 * ery * b
            m[int(max(0, yline)):, :] = 0
            img[..., :3] = img[..., :3] * (1 - m[..., None]) + src[..., :3] * m[..., None]
            pts = np.array([[ex - erx + 4, yline - 6 * self.s], [ex, yline + (6 + 8 * b) * self.s], [ex + erx - 4, yline - 8 * self.s]], np.float32)
            curve = np.array([(1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t * t * pts[2] for t in np.linspace(0, 1, 24)], np.int32)
            lash = np.zeros((h, w), np.float32)
            cv2.polylines(lash, [curve], False, 1.0, max(2, int(self.c["lash_w"] * self.s)), cv2.LINE_AA)
            lash = cv2.GaussianBlur(lash, (0, 0), 1.2)
            lc = np.array(self.c["lash"][::-1], np.float32) / 255
            img[..., :3] = img[..., :3] * (1 - lash[..., None]) + lc * lash[..., None]


def over(dst, src, x, y):
    """Alpha-composite BGRA float src onto BGR float dst at (x, y)."""
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4]
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a


def background(kind, t, beat):
    if kind == "pepper":
        if not hasattr(background, "pep"):
            bg = cv2.imread(f"{ROOT}/public/paint/pepper-bg.jpg")
            background.pep = cv2.resize(bg, (W, H)).astype(np.float32) / 255
        img = background.pep * (0.72 + 0.18 * beat)
    else:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        img = np.broadcast_to((0.10 + 0.10 * yy) * np.array([1.2, 0.9, 1.0], np.float32), (H, W, 3)).copy()
    # soft stage lights (bokeh), one of them pulsing per bar
    if not hasattr(background, "blobs"):
        background.blobs = []
        for (x, y, r, col) in [(300, 300, 260, (0.76, 0.84, 0.50)), (1620, 260, 240, (0.42, 0.70, 0.91)), (520, 820, 220, (0.88, 0.55, 0.65)), (1420, 780, 260, (0.55, 0.48, 0.88))]:
            m = np.zeros((H, W), np.float32)
            cv2.circle(m, (x, y), r, 1.0, -1)
            background.blobs.append((cv2.GaussianBlur(m, (0, 0), r * 0.35)[..., None], np.array(col, np.float32)))
    bar = int(t / 2.0) % 4
    for i, (m, col) in enumerate(background.blobs):
        img = img + m * col * (0.10 + (0.22 * beat if i == bar else 0))
    return img


def render(name, start=8.0, dur=15.0):
    cfg = CHARS[name]
    p = Puppet(cfg)
    n = int(dur * FPS)
    env, vowels = drivers(start, n)
    hb, wb = p.body.shape[:2]
    x0 = (W - wb) // 2
    out = f"{ROOT}/out/warp-{name}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-ss", str(start), "-t", str(dur), "-i", f"{ROOT}/public/song/song.wav",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out], stdin=subprocess.PIPE)
    rng = np.random.default_rng(1)
    vign = None
    prev_turn = 0.0
    for f in range(n):
        t = start + f / FPS
        ph = (t % 0.5) / 0.5
        beat = math.exp(-ph * 6)  # 1 on the beat, fading
        on_beat = math.cos(math.pi * ph) ** 2
        turn = 0.75 * smooth_noise(t, 1)
        tilt = 3.0 * smooth_noise(t, 4) + 1.2 * on_beat
        nod = 7 * cfg["height"] / 1500 * on_beat
        sway = (prev_turn - turn) * 260 * p.s + 4 * p.s * math.sin(t * 2.3)
        prev_turn = turn
        k = int(t / 3.3)
        bt = t - (k * 3.3 + (k * 7 % 10) / 11)
        blink = max(0.0, 1 - abs(bt - 0.07) / 0.07) if 0 <= bt < 0.14 else 0.0
        head = p.head_frame(turn, float(env[f]), vowels[f], blink, sway)
        # tilt + nod around the neck pivot
        M = cv2.getRotationMatrix2D((p.pivot[0], p.pivot[1]), -tilt, 1.0)
        M[1, 2] += nod
        head = cv2.warpAffine(head, M, (head.shape[1], head.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
        breathe = 1 + 0.006 * math.sin(t * 1.7)
        body = cv2.resize(p.body, (wb, int(hb * breathe)))
        frame = background(cfg["bg"], t, beat)
        yb = cfg["top"] + hb - body.shape[0] + int(3 * math.sin(t * 1.7))
        over(frame, body, x0, yb)
        over(frame, head, x0, cfg["top"] + int(3 * math.sin(t * 1.7)))
        # camera: slow push + tiny drift
        z = 1 + 0.06 * f / n
        C = cv2.getRotationMatrix2D((W / 2 + 8 * math.sin(t * 0.8), H * 0.42), 0, z)
        frame = cv2.warpAffine(frame, C, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if vign is None:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            vign = (1 - 0.45 * np.clip(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2, 0, 1))[..., None]
        frame = frame * vign + rng.normal(0, 0.018, (H, W, 1)).astype(np.float32)
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        cv2.putText(frame, cfg["credit"], (30, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    render(sys.argv[1], *(float(a) for a in sys.argv[2:]))
