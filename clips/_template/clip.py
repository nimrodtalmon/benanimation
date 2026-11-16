"""A new clip: copy this folder to clips/<name>/ and edit. See docs/GUIDE.md.

  python3 clips/<name>/clip.py stills 3 16 24.5     # check frames first
  python3 tools/prebuild.py clips/<name>/clip.py     # build the 3D close-ups ahead (optional)
  python3 clips/<name>/clip.py video                 # -> out/<name>.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from scenery import ben_logo  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402
from venues import Rooftop  # noqa: E402  (or Club, Campfire)

BAND = [Member("drums", "distor"),                                  # rigs in public/rigs/ (+ public/parts/)
        Member("guitar", "blonde", side="blonde_side", sings=False),
        Member("bass", "bassboy", seed=1)]
show = Show(Song.load("dr1"), BAND, name="template", script=__file__, logo=ben_logo(),
            venue=Rooftop(sky="tour"),    # where; sky: twilight / stars / clouds / aurora / a schedule
            style3d="ben",                # everything 3D drawn like the drawings (None: photoreal 3D)
            closeups=5, moves=True,       # 3D hands on instruments; camera moves
            end_lines=["music: ...", "drawings: Ben"])
prod = show.production()

if __name__ == "__main__":
    prod.main(sys.argv)
