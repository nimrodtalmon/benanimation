"""Clip 'rooftop': the band (as clips/distorted) on a city rooftop at night, everything 3D drawn
like Ben's drawings (style3d="ben"). The city lights up on the beat through the intro, blacks out
in the breaks, comes back on the shouts.

  python3 clips/rooftop/clip.py stills 10 30 50
  python3 clips/rooftop/clip.py video
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from scenery import ben_logo  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402
from venues import Rooftop  # noqa: E402

BAND = [Member("drums", "distor"), Member("guitar", "blonde", side="blonde_side"), Member("bass", "bassboy", seed=1)]
show = Show(Song.load("dr1"), BAND, name="rooftop", script=__file__, logo=ben_logo(), venue=Rooftop(), style3d="ben",
            end_lines=["music: Distorted Reality", "drawings: Ben"])
prod = show.production()

if __name__ == "__main__":
    prod.main(sys.argv)
