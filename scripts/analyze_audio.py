"""Analyze a song for the edit: tempo, beats, sections, and per-frame loudness curves.

  pip install librosa
  python scripts/analyze_audio.py SONG.(mp4|m4a|wav) src/clip-analysis.json

Output curves are per video frame (30 fps), scaled to 0..1:
  rms (overall), bass (30-150 Hz), mid (300-3000 Hz: voice, guitars), onset (hits).
"""
import json
import subprocess
import sys
import tempfile

import librosa
import numpy as np

FPS = 30


def main(src, out):
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "22050", wav.name], check=True)
        y, sr = librosa.load(wav.name, sr=22050)
    hop = int(sr / FPS)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band(lo, hi):
        return S[(f >= lo) & (f < hi)].mean(0)

    def norm(a):
        return np.round((a / (np.percentile(a, 98) + 1e-9)).clip(0, 1), 3).tolist()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    bounds = librosa.segment.agglomerative(mfcc, 8)
    data = {
        "fps": FPS,
        "duration": len(y) / sr,
        "tempo": float(np.atleast_1d(tempo)[0]),
        "beats": [round(float(b), 3) for b in librosa.frames_to_time(beats, sr=sr)],
        "rms": norm(librosa.feature.rms(y=y, hop_length=hop)[0]),
        "bass": norm(band(30, 150)),
        "mid": norm(band(300, 3000)),
        "onset": norm(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)),
        "segments": [round(float(b) / FPS, 2) for b in bounds],
    }
    json.dump(data, open(out, "w"))
    print(f"tempo {data['tempo']:.1f}, {len(data['beats'])} beats, sections at {data['segments']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
