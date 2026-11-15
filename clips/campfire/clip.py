"""Clip 'campfire': the same band and song as clips/distorted, somewhere else. The bassist walks
through the woods at sunset playing the intro, the guitarist walks with him; when the band comes
in they reach a clearing where the drummer sits at his kit by a fire, which flares up. The fire
plays along (embers in the breaks, flare-ups on the shouts, sparks on the crashes) and dies down
at the end. Only the venue differs from clips/distorted (puppet/venues.py: Campfire).

  python3 clips/campfire/clip.py stills 3 16 40
  python3 clips/campfire/clip.py video               -> out/campfire.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from scenery import ben_logo  # noqa: E402
from show import Member, Show  # noqa: E402
from song import Song  # noqa: E402
from venues import Campfire  # noqa: E402

BAND = [Member("drums", "distor"), Member("guitar", "blonde", side="blonde_side"), Member("bass", "bassboy", seed=1)]
show = Show(Song.load("dr1"), BAND, name="campfire", script=__file__, logo=ben_logo(), venue=Campfire(),
            end_lines=["music: Distorted Reality", "drawings: Ben"])
prod = show.production()

if __name__ == "__main__":
    prod.main(sys.argv)
