"""Clip 1: the band (Distor sings, Pepper drums, Coriander plays guitar) to 'silence in the air'.

  python3 clips/silence/clip.py video              -> out/silence.mp4 (renders 2 halves in parallel)
  python3 clips/silence/clip.py stills 3 15 21     -> out/silence-still_<t>.jpg
  python3 clips/silence/clip.py singer 8 6         -> one character alone (singer|drummer|guitarist)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.coriander import Guitarist  # noqa: E402
from cast.distor import Singer  # noqa: E402
from cast.pepper import Drummer  # noqa: E402
from production import Production, W  # noqa: E402
from song import Song  # noqa: E402

song = Song.default()

CAST = {
    "singer": lambda: Singer(song),
    "singer_close": lambda: Singer(song, scale=0.62, top=-10),
    "singer_full": lambda: Singer(song, scale=0.245, top=40),
    "drummer": lambda: Drummer(song),
    "guitarist": lambda: Guitarist(song),
    # the wide shot: three smaller copies placed across the stage
    "w_drummer": lambda: Drummer(song, scale=0.40, top=150),
    "w_singer": lambda: Singer(song, scale=0.28, cx=W * 0.24, top=470),
    "w_guitarist": lambda: Guitarist(song, scale=0.42, cx=W * 0.76, top=int(330 - 120 * 0.42)),
}
WIDE = ("w_drummer", "w_singer", "w_guitarist")

# a cut on every bar line (a bar = 2 s at 120 bpm)
SHOTS = [(0, 2, "guitarist"), (2, 4, WIDE), (4, 6, "drummer"), (6, 8, "guitarist"), (8, 10, WIDE),
         (10, 14, "singer"), (14, 16, "singer_close"), (16, 18, "guitarist"), (18, 20, "singer_full"), (20, 22, WIDE),
         (22, 24, "guitarist"), (24, 26, "drummer"), (26, 28, "singer"), (28, 30, "singer_full"), (30, 32, "guitarist"),
         (32, 34, "drummer"), (34, 38, "singer_close"), (38, 40, WIDE), (40, 42, "drummer"), (42, 44, WIDE)]

prod = Production(song, CAST, SHOTS, name="silence", script=os.path.abspath(__file__))

if __name__ == "__main__":
    prod.main(sys.argv)
