"""A song for a clip: the score (drums, strums, sung notes, mouth shapes; a JSON like
src/song.json) plus the audio and an aligned vocal-only track. Characters read every timing
from here, so a new clip only needs a new Song.

  song = Song("src/song.json", audio="public/song/song.wav", vocals="public/song/vocals.wav",
              credit="Song: ... (license)")
"""
import json
import math
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30


class Song:
    def __init__(self, score, audio, vocals=None, credit="", root=ROOT):
        p = lambda x: x if os.path.isabs(x) else f"{root}/{x}"  # noqa: E731
        self.data = json.load(open(p(score)))
        self.audio = p(audio)
        self.vocals = p(vocals) if vocals else None
        self.credit = credit
        self.bpm = self.data["bpm"]
        self.spb = 60 / self.bpm
        self.bar = 4 * self.spb
        self.duration = self.data["duration"]
        self._drv = None

    @classmethod
    def default(cls):
        return cls("src/song.json", "public/song/song.wav", "public/song/vocals.wav",
                   "Song: 'silence in the air' by JeffSpeed68 (ccmixter.org, CC BY-NC 3.0) + our drums")

    # ---- a pre-roll (a scene before the music)
    def shifted(self, pre, name="pre", count_in=None, sfx=()):
        """A copy of this song that starts `pre` seconds later (pre: whole bars keeps the beat
        grid). count_in=(t0, n): n closed hi-hat ticks one beat apart from t0, added to the score
        (so a drummer plays them) and to the audio. sfx: [(t, "step"|"tick", gain)] extra sounds
        (footsteps...). Audio goes to out/<name>-song.wav and out/<name>-vocals.wav."""
        import copy
        sr = 44100
        new = copy.copy(self)
        d = copy.deepcopy(self.data)
        for key in ("drums", "vocal", "guitar"):
            for e in d.get(key, []):
                e["t"] += pre
        for sec in d.get("sections", []):
            sec["t0"] += pre
            sec["t1"] += pre
        d["sections"] = [dict(name="pre", t0=0.0, t1=pre)] + d.get("sections", [])
        d["duration"] = self.duration + pre
        sounds = list(sfx)
        if count_in:
            t0, n = count_in
            for k in range(n):
                tk = t0 + k * self.spb
                d["drums"].append(dict(beat=None, piece=42, vel=70, t=tk))
                sounds.append((tk, "tick", 0.9))
        d["drums"].sort(key=lambda e: e["t"])
        new.data, new.duration, new._drv = d, d["duration"], None

        def load(path):
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"], capture_output=True).stdout
            return np.frombuffer(raw, np.float32).reshape(-1, 2)

        def save(y, path):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(sr), "-i", "-", path],
                           input=np.clip(y, -1, 1).astype(np.float32).tobytes(), check=True)

        rng = np.random.default_rng(5)
        os.makedirs(f"{ROOT}/out", exist_ok=True)
        y = load(self.audio)
        y = np.concatenate([np.zeros((int(pre * sr), 2), np.float32), y])
        y[: int(pre * sr)] += 0.003 * rng.normal(size=(int(pre * sr), 2)).astype(np.float32)  # room tone
        for (ts, kind, g) in sounds:
            i = int(ts * sr)
            if kind == "tick":  # closed hi-hat: bright noise, ~40 ms
                n = int(0.05 * sr)
                z = rng.normal(size=n)
                z = z - np.convolve(z, np.ones(4) / 4, "same")
                w = z * np.exp(-np.arange(n) / (0.012 * sr))
            else:  # a footstep: a soft low thump plus a little scuff
                n = int(0.12 * sr)
                tt = np.arange(n) / sr
                w = np.sin(2 * np.pi * 70 * tt) * np.exp(-tt / 0.025) + 0.15 * rng.normal(size=n) * np.exp(-tt / 0.01)
            w = (g * 0.5 * w / (np.abs(w).max() + 1e-9)).astype(np.float32)
            j = min(len(y), i + len(w))
            if 0 <= i < len(y):
                y[i:j] += w[: j - i, None]
        new.audio = f"{ROOT}/out/{name}-song.wav"
        save(y, new.audio)
        if self.vocals:
            v = load(self.vocals)
            new.vocals = f"{ROOT}/out/{name}-vocals.wav"
            save(np.concatenate([np.zeros((int(pre * sr), 2), np.float32), v]), new.vocals)
        return new

    # ---- the voice
    def vocal_env(self):
        """Loudness of the vocal track per frame, 0..1 (fast attack, ~70 ms release)."""
        sr = 16000
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", self.vocals, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                             capture_output=True).stdout
        y = np.frombuffer(raw, np.float32)
        hop = sr // FPS
        rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(len(y) // hop)])
        rms /= np.percentile(rms, 95)
        env = np.zeros_like(rms)
        for i in range(len(rms)):
            prev = env[i - 1] if i else 0
            env[i] = rms[i] if rms[i] > prev else prev * 0.62 + rms[i] * 0.38
        return np.clip((env - 0.12) / 0.88, 0, 1) ** 0.8

    def vocal_drivers(self):
        """(loudness, vowel, singing-flag) per frame for the whole song; the flag looks 0.6 s
        ahead and bridges pauses under 1.4 s (so a mic goes up before the line, not after)."""
        if self._drv is not None:
            return self._drv
        n = int(self.duration * FPS)
        env = self.vocal_env() if self.vocals else np.zeros(n)
        env = np.concatenate([env, np.zeros(max(0, n - len(env)))])[:n]
        vowels = ["ah"] * n
        for c in self.data.get("vocal", []):
            for f in range(int(c["t"] * FPS), min(n, int((c["t"] + c["dur"] + 0.05) * FPS) + 1)):
                vowels[f] = c["vowel"]
        look = np.array([env[max(0, i - 9):i + 18].max() for i in range(n)])
        sing = look > 0.25
        idx = np.nonzero(sing)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a < 1.4 * FPS:
                sing[a:b] = True
        self._drv = (env, vowels, sing.astype(float))
        return self._drv

    def vocal_accents(self):
        """Onsets of loud syllables (time, strength): where a singer nods."""
        env, _, _ = self.vocal_drivers()
        out, last = [], -9
        for i in range(2, len(env)):
            if env[i] > 0.55 and env[i - 2] < 0.3 and i - last > 0.35 * FPS:
                out.append((i / FPS, 0.6 + 0.4 * env[i]))
                last = i
        return out

    # ---- the beat
    def beat_phase(self, t):
        return (t % self.spb) / self.spb

    def beat_pulse(self, t, k=5.0):
        return math.exp(-self.beat_phase(t) * k)
