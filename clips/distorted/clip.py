"""Clip 5 'distorted': the band Distorted Reality (Ben's characters) plays their own song (an
instrumental, recorded by the band). Everything comes from puppet/show.py (the band show recipe):
this file only says which song, which band, and the end card.

  python3 clips/distorted/clip.py segment 13 23      -> a short piece with audio
  python3 clips/distorted/clip.py stills 2 16 33
  python3 clips/distorted/clip.py video               -> out/distorted.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from scenery import ben_logo  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402

BAND = [Member("drums", "distor"), Member("guitar", "blonde", side="blonde_side"), Member("bass", "bassboy", seed=1)]
show = Show(Song.load("dr1"), BAND, name="distorted", script=__file__, logo=ben_logo(),
            end_lines=["music: Distorted Reality", "drawings: Ben"])
prod = show.production()

if __name__ == "__main__":
    prod.main(sys.argv)
