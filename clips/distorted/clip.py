"""Clip 5 'distorted': the band Distorted Reality (Ben's characters) plays their own song (an
instrumental). Distor on drums; the guitarist and bassist come next.

  python3 clips/distorted/clip.py segment 16 26      -> a short piece with audio
  python3 clips/distorted/clip.py stills 20 30
  python3 clips/distorted/clip.py video
Song: public/songs/dr1 (score from puppet/transcribe.py; stems from Demucs).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.distor_drums import DistorDrummer  # noqa: E402
from production import HandDrawn, Looks, Production, TVStatic  # noqa: E402
from song import Song  # noqa: E402

song = Song("public/songs/dr1/score.json", "public/songs/dr1/song.mp3", None, "Music: Distorted Reality (the band)")

HEAD = 0.6
CAST = {
    "drums": lambda: DistorDrummer(song, head=HEAD, face=True),
    "drums_close": lambda: DistorDrummer(song, scale=0.85, top=-60, head=HEAD, face=True),
}
SHOTS = [(0, 99, "drums")]
CRASH = np.array([d["t"] for d in song.data["drums"] if d["piece"] == 49])

# the TV look, toned down (round-3 feedback: too much); kept as a dial
prod = Production(song, CAST, SHOTS, name="distorted", script=os.path.abspath(__file__),
                  look=Looks(HandDrawn(), TVStatic(0.05, vhs=0.25)), stage="stage", crowd_wide=True,
                  drift=0.065, parallax=0.55, ghost=0.35, punch=(CRASH, 0.035), roll=0.5)

if __name__ == "__main__":
    prod.main(sys.argv)
