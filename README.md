# benanimation

Gorillaz-style music clips from Ben's hand-drawn characters. Everything is code: drawings + a song in, MP4 out.

- **Start here: `docs/GUIDE.md`** — how to make a clip from new drawings, a song and a story.
- `docs/STATUS.md` — where things stand, open issues, and the history of each round.
- `docs/TOOLKIT.md` — the pieces in detail and lessons learned.
- `docs/BEN_DRAWING_KIT.md` — what Ben should draw (upload app: `ben-app/`, live at talmonlab.bgu.ac.il/ben/).

```
python3 clips/rooftop/clip.py stills 16 24.5     # a few frames
python3 clips/rooftop/clip.py video              # the whole clip -> out/rooftop.mp4
cp -r clips/_template clips/<new>                # a new clip
python3 tools/smoke.py                           # quick check after code changes
```

Layout: `puppet/` (the library), `clips/` (one script per clip; old ones in `clips/legacy/`), `public/` (rigs, parts,
songs, instrument/kit/scenery specs), `drawings/` (scans), `tools/`, `legacy/remotion/` (the first, React version).
Renders and caches go to `out/` (not committed).

Credits: songs keep their credits in the videos ("silence in the air" by JeffSpeed68, CC BY-NC 3.0); MakeHuman hand
mesh (CC0, `public/hands/README.md`); David Revoy's Pepper & Carrot art in legacy clips (CC BY 4.0).
