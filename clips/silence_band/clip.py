"""Clip 6 'silence_band': Ben's band (Distorted Reality's characters) plays 'silence in the air'
(JeffSpeed68, CC BY-NC 3.0), the blonde singing. The first clip made only with the recipe:
puppet/new_song.py brought the song in, puppet/show.py builds the whole show.

  python3 clips/silence_band/clip.py stills 3 12 30
  python3 clips/silence_band/clip.py video          -> out/silence_band.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from scenery import ben_logo  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402

BAND = [Member("drums", "distor"), Member("guitar", "blonde", sings=True, side="blonde_side"), Member("bass", "bassboy", seed=1)]
show = Show(Song.load("silence"), BAND, name="silence_band", script=__file__, logo=ben_logo(),
            end_lines=["song: 'silence in the air' by JeffSpeed68 (CC BY-NC 3.0)", "drawings: Ben"])
prod = show.production()

if __name__ == "__main__":
    prod.main(sys.argv)
