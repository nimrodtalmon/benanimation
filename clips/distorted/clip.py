"""Clip 5 'distorted': the band Distorted Reality (Ben's characters) plays their own song (an
instrumental, recorded by the band). Distor on drums, the blonde on guitar, the boy from the band
page on bass.

  python3 clips/distorted/clip.py segment 16 26      -> a short piece with audio
  python3 clips/distorted/clip.py stills 5 20 30
  python3 clips/distorted/clip.py video               -> out/distorted.mp4
Song: public/songs/dr1 (score from puppet/transcribe.py on Demucs stems).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.bust_player import BustPlayer  # noqa: E402
from cast.distor_drums import DistorDrummer  # noqa: E402
from production import W, HandDrawn, Looks, Production, TVStatic  # noqa: E402
from song import Song  # noqa: E402

song = Song("public/songs/dr1/score.json", "public/songs/dr1/song.mp3", None, "Music: Distorted Reality (the band)")

HEAD = 0.6
CAST = {
    "drums": lambda: DistorDrummer(song, head=HEAD, face=True),
    "drums_close": lambda: DistorDrummer(song, scale=0.85, top=-60, head=HEAD, face=True),
    "guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.40, top=30, head=HEAD, seed=3),
    "guitar_close": lambda: BustPlayer(song, "blonde", "guitar", scale=0.62, top=-60, head=HEAD, seed=3),
    "bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.55, top=40, head=HEAD, seed=5),
    "bass_close": lambda: BustPlayer(song, "bassboy", "bass", scale=0.80, top=-90, head=HEAD, seed=5),
    # the wide shot: drums at the back, guitar and bass in front, bodies running off the bottom
    "w_drums": lambda: DistorDrummer(song, scale=0.36, top=150, head=HEAD, face=True),
    "w_guitar": lambda: BustPlayer(song, "blonde", "guitar", scale=0.25, cx=W * 0.24, top=400, head=HEAD, seed=3),
    "w_bass": lambda: BustPlayer(song, "bassboy", "bass", scale=0.34, cx=W * 0.77, top=330, head=HEAD, seed=5),
}
WIDE = ("w_drums", "w_guitar", "w_bass")


def shot_list():
    """A cut on every bar line (bars from the beat grid, 4 beats each); two bars per shot in the
    intro (bass alone), and a fixed cycle of angles after the drums come in."""
    beats = song.data["beats"]
    bars = beats[::4] + [song.duration + 1]
    drums_in = min(d["t"] for d in song.data["drums"] if d["t"] > 10)
    intro = ["bass", WIDE, "bass_close", "bass", WIDE, "bass_close", "bass"]
    cycle = ["drums", WIDE, "guitar", "bass", "drums_close", "guitar_close", WIDE, "bass_close", "drums", "guitar", WIDE, "bass"]
    shots, i, k, kc = [], 0, 0, 0
    t0 = 0.0
    while i < len(bars) - 1:
        if bars[i] < drums_in - 0.3:  # intro: two bars per shot
            j = min(i + 2, len(bars) - 1)
            while bars[j] > drums_in - 0.3 and j > i + 1:
                j -= 1
            who = intro[k % len(intro)]
        else:
            j = i + 1
            who = cycle[kc % len(cycle)]  # the drums come in on a drums shot
            kc += 1
        shots.append((t0, bars[j], who))
        t0, i, k = bars[j], j, k + 1
    return shots


SHOTS = shot_list()
CRASH = np.array([d["t"] for d in song.data["drums"] if d["piece"] == 49])

# the TV look, toned down (round-3 feedback: too much); kept as a dial
prod = Production(song, CAST, SHOTS, name="distorted", script=os.path.abspath(__file__),
                  look=Looks(HandDrawn(), TVStatic(0.05, vhs=0.25)), stage="stage", crowd_wide=False,
                  drift=0.065, parallax=0.55, ghost=0.35, punch=(CRASH, 0.035), roll=0.5)

if __name__ == "__main__":
    prod.main(sys.argv)
