"""Clip 4 'band2': clip 1's story (the band plays the whole song) in the style picked from style reel round 2:
colored stage lights (C) + worn-tape TV (D) + one cut per bar (E), faces on the drummer and guitarist,
and a bit more camera than E (drift, parallax, frame blending, small kicks on crashes, slight roll).

  python3 clips/band2/clip.py video              -> out/band2.mp4 (2 halves in parallel)
  python3 clips/band2/clip.py stills 3 15 21     -> out/band2-still_<t>.jpg
  python3 clips/band2/clip.py segment 18 28      -> a short piece with audio
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.coriander import Guitarist  # noqa: E402
from cast.distor import Singer  # noqa: E402
from cast.pepper import Drummer  # noqa: E402
from production import W, HandDrawn, Looks, Production, TVStatic  # noqa: E402
from song import Song  # noqa: E402

song = Song.default()

HEAD = 0.6  # calm heads (Clint Eastwood)
SING = dict(fluid=True, head=HEAD, mouth_twos=True)
CAST = {
    "singer": lambda: Singer(song, **SING),
    "singer_close": lambda: Singer(song, scale=0.62, top=-10, **SING),
    "singer_full": lambda: Singer(song, scale=0.245, top=40, **SING),
    "drummer": lambda: Drummer(song, head=HEAD, face=True),
    "drummer_close": lambda: Drummer(song, scale=0.85, top=-25, head=HEAD, face=True),
    "guitarist": lambda: Guitarist(song, head=HEAD, face=True),
    "w_drummer": lambda: Drummer(song, scale=0.40, top=150, head=HEAD, face=True),
    "w_singer": lambda: Singer(song, scale=0.28, cx=W * 0.24, top=470, **SING),
    "w_guitarist": lambda: Guitarist(song, scale=0.42, cx=W * 0.76, top=int(330 - 120 * 0.42), head=HEAD, face=True),
}
WIDE = ("w_drummer", "w_singer", "w_guitarist")

# clip 1's shot list: a cut on every bar line (a bar = 2 s at 120 bpm); one drummer close-up added
SHOTS = [(0, 2, "guitarist"), (2, 4, WIDE), (4, 6, "drummer"), (6, 8, "guitarist"), (8, 10, WIDE),
         (10, 14, "singer"), (14, 16, "singer_close"), (16, 18, "guitarist"), (18, 20, "singer_full"), (20, 22, WIDE),
         (22, 24, "guitarist"), (24, 26, "drummer"), (26, 28, "singer"), (28, 30, "singer_full"), (30, 32, "guitarist"),
         (32, 34, "drummer_close"), (34, 38, "singer_close"), (38, 40, WIDE), (40, 42, "drummer"), (42, 44, WIDE)]

CRASH = np.array([d["t"] for d in song.data["drums"] if d["piece"] == 49])

prod = Production(song, CAST, SHOTS, name="band2", script=os.path.abspath(__file__),
                  look=Looks(HandDrawn(), TVStatic(0.14, vhs=0.8)), stage="stage", crowd_wide=True,
                  drift=0.065, parallax=0.55, ghost=0.35, punch=(CRASH, 0.035), roll=0.5)

if __name__ == "__main__":
    prod.main(sys.argv)
