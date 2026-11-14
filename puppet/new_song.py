"""Bring a new song into the toolkit, in one step:

  python3 puppet/new_song.py <audio file> <id> --credit "Music: ... (license)"

-> public/songs/<id>/song.mp3, vocals.mp3 (the voice alone, if there is one) and score.json
   (drums, bass, guitar, vocal syllables, beats, sections, stop-time breaks), ready for
   Song.load(<id>) and puppet/show.py. The stems (Demucs, CPU, a few minutes) are cached in
   out/stems/<id>/. Check the score by eye with tools/score_plot.py.
"""
import json
import os
import subprocess
import sys

import numpy as np

import transcribe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(audio, sid, credit=""):
    dst = f"{ROOT}/public/songs/{sid}"
    os.makedirs(dst, exist_ok=True)
    work = f"{ROOT}/out/stems/{sid}"
    os.makedirs(work, exist_ok=True)
    wav = f"{work}/mix.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", audio, "-ar", "44100", "-ac", "2", wav], check=True)
    stems = f"{work}/htdemucs/mix"
    if not os.path.exists(f"{stems}/drums.wav"):
        subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", "cpu", "-o", work, wav], check=True)
    score = transcribe.transcribe(wav, stems, credit)
    json.dump(score, open(f"{dst}/score.json", "w"), indent=0)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", "libmp3lame", "-b:a", "192k", f"{dst}/song.mp3"], check=True)
    if score["vocal"]:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{stems}/vocals.wav", "-c:a", "libmp3lame", "-b:a", "128k", f"{dst}/vocals.mp3"], check=True)
    json.dump(dict(credit=credit, vocals=bool(score["vocal"])), open(f"{dst}/song.json", "w"), indent=1)
    from collections import Counter
    print(dst, "bpm", score["bpm"], "dur", score["duration"], "drums", dict(Counter(d["piece"] for d in score["drums"])),
          "bass", len(score["bass"]), "guitar", len(score["guitar"]), "vocal", len(score["vocal"]), "breaks", len(score["holes"]))


if __name__ == "__main__":
    a = sys.argv[1:]
    credit = ""
    if "--credit" in a:
        i = a.index("--credit")
        credit = a[i + 1]
        a = a[:i] + a[i + 2:]
    main(a[0], a[1], credit)
