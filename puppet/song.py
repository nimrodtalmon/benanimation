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
