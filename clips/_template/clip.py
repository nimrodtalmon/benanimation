"""Template for a new clip. Copy this folder to clips/<name>/ and edit.

1. A Song: a score JSON (bpm, duration, and whatever events your characters use: "drums",
   "guitar", "vocal" ...), the mixed audio and (for singing) an aligned vocal-only track.
2. A cast: name -> factory. Characters live in puppet/cast/<name>.py; each has
   __init__(song, scale, cx, top), draw(frame, t), and j["credit"]. A new character = one module.
3. SHOTS: (start, end, name) or (start, end, (name, name, ...)) for several in one frame.

  python3 clips/<name>/clip.py stills 1 3 5     -> check frames first
  python3 clips/<name>/clip.py video            -> out/<name>.mp4
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.distor import Singer  # noqa: E402
from production import Production  # noqa: E402
from song import Song  # noqa: E402

song = Song.default()  # replace: Song("clips/<name>/score.json", "clips/<name>/mix.wav", "clips/<name>/vocals.wav", "Song: ... (license)")

CAST = {
    "hero": lambda: Singer(song),
    "hero_full": lambda: Singer(song, scale=0.245, top=40),
}
SHOTS = [(0, 4, "hero_full"), (4, song.duration, "hero")]

prod = Production(song, CAST, SHOTS, name="template", script=os.path.abspath(__file__))

if __name__ == "__main__":
    prod.main(sys.argv)
