"""Test song with a real singer: an excerpt of "silence in the air" by JeffSpeed68
(Stefan Kartenberg), ccMixter, CC BY-NC 3.0, plus our own drums on its 120 BPM grid.

Inputs (not in git): STEMS/mix.mp3 and STEMS/vocals.flac from
https://ccmixter.org/files/JeffSpeed68/53352, and the Rhubarb Lip Sync binary.

  python music/song2.py /home/claude/stems     # -> public/song/song.wav, src/song.json

The JSON has the same shape as music/score.py's, so the band code is unchanged:
drums = our pattern; guitar = strums detected in the mix on the 8th-note grid;
vocal = mouth shapes from Rhubarb Lip Sync run on the vocal stem.
"""
import json
import os
import subprocess
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import score  # noqa: E402

START, LENGTH = 14.02, 44.0  # excerpt in the original (first beat of a bar, 22 bars at 120 BPM)
VOX_OFFSET = 6.0  # the vocal stem starts 6.0 s into the full mix (measured by cross-correlation)
# Rhubarb mouth shapes -> our vowels and how open
SHAPES = {"B": ("ee", 0.35), "C": ("ah", 0.6), "D": ("ah", 1.0), "E": ("oh", 0.75), "F": ("oo", 0.6), "G": ("ee", 0.25), "H": ("ah", 0.5)}


def main(stems):
    sh = lambda *a: subprocess.run(list(a), check=True, capture_output=True)  # noqa: E731
    sh("ffmpeg", "-y", "-ss", str(START - VOX_OFFSET), "-t", str(LENGTH), "-i", f"{stems}/vocals.flac", "-ac", "1", "-ar", "16000", "/tmp/vox.wav")
    sh("ffmpeg", "-y", "-ss", str(START), "-t", str(LENGTH), "-i", f"{stems}/mix.mp3", "-ar", "44100", "/tmp/mix.wav")
    sh(f"{stems}/Rhubarb-Lip-Sync-1.14.0-Linux/rhubarb", "-f", "json", "--extendedShapes", "GHX", "-r", "phonetic", "-o", "/tmp/cues.json", "/tmp/vox.wav")

    # our drums: intro 2 bars, verse 8, chorus 10, outro 2
    score.SECTIONS = [("intro", 2), ("verse", 8), ("chorus", 10), ("outro", 2)]
    ev = score.build()
    drums_only = {**ev, "guitar": [], "bass": [], "vocal": []}
    score.to_midi(drums_only, "/tmp/drums.mid")
    sh("fluidsynth", "-ni", "-g", "0.9", "-r", "44100", "-F", "/tmp/drums.wav", "/usr/share/sounds/sf2/FluidR3_GM.sf2", "/tmp/drums.mid")
    sh("ffmpeg", "-y", "-i", "/tmp/mix.wav", "-i", "/tmp/drums.wav", "-filter_complex",
       f"[1:a]volume=0.6[d];[0:a][d]amix=inputs=2:duration=first:normalize=0,afade=t=out:st={LENGTH - 2.5}:d=2.5,loudnorm=I=-14:TP=-1.5",
       "-ar", "44100", "public/song/song.wav")

    out = {"bpm": 120, "duration": LENGTH, "sections": ev["sections"], "bass": [],
           "credit": "Song: 'silence in the air' by JeffSpeed68 (Stefan Kartenberg), ccMixter, CC BY-NC 3.0"}
    out["drums"] = [{**d, "t": round(d["beat"] * 0.5, 4)} for d in ev["drums"]]

    # sung "notes" from mouth shapes (merge repeats; A/X = closed)
    vocal = []
    for c in json.load(open("/tmp/cues.json"))["mouthCues"]:
        if c["value"] in ("A", "X"):
            continue
        vowel, op = SHAPES[c["value"]]
        if vocal and vocal[-1]["vowel"] == vowel and abs(vocal[-1]["t"] + vocal[-1]["dur"] - c["start"]) < 1e-3:
            vocal[-1]["dur"] = round(c["end"] - vocal[-1]["t"], 3)
        else:
            vocal.append({"t": c["start"], "dur": round(c["end"] - c["start"], 3), "vowel": vowel, "open": op, "pitch": 66})
    out["vocal"] = vocal

    # guitar strums: onsets in the mix (mid band) on the 8th-note grid; chord root from chroma per bar
    y, sr = librosa.load("/tmp/mix.wav", sr=22050)
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = S[(f > 200) & (f < 2500)].sum(0)
    on = np.maximum(0, np.diff(band, prepend=band[0]))
    on /= np.percentile(on, 99) + 1e-9
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    roots = {"Em": 4, "G": 7, "C": 0, "D": 2}
    guitar = []
    for k in range(int(LENGTH / 0.25)):
        t = k * 0.25
        i = int(t * sr / hop)
        strength = on[max(0, i - 3): i + 4].max() if i < len(on) else 0
        bar = int(t // 2)
        b0, b1 = int(bar * 2 * sr / hop), int((bar + 1) * 2 * sr / hop)
        chord = max(roots, key=lambda c: chroma[roots[c], b0:b1].mean()) if b1 <= chroma.shape[1] else "Em"
        if strength > 0.18 or k % 4 == 0:
            guitar.append({"t": t, "dur": 0.24, "root": 40, "chord": chord, "dir": "down" if k % 2 == 0 else "up", "accent": k % 8 == 0})
    out["guitar"] = guitar
    json.dump(out, open("src/song.json", "w"))
    print(f"{len(vocal)} sung shapes, {len(guitar)} strums, {len(out['drums'])} drum hits")


if __name__ == "__main__":
    main(sys.argv[1])
