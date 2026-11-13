# benanimation

Animated clips from Ben's hand-drawn characters. Everything is code: scans in, MP4 out.

## Warp-puppet band (current direction)

Start with **docs/TOOLKIT.md**: the reusable pipeline, how to add a character, and lessons learned.


Everything in Python + OpenCV (`puppet/`), every move timed to the song (`src/song.json` + `public/song/vocals.wav`):

- `puppet/warp.py`: one drawing + landmarks -> head turn, jaw/mouth, blinks, hair sway, tilt/nod (Live2D-style pixel warps).
- `puppet/limb.py`: bending arms from Ben's two arm drawings (upper arm, forearm+hand) or one painted arm: 2-bone IK,
  rigid parts, the outer elbow swept from the drawing's own cross-section, forearm cap as the crease, foreshortening.
- `puppet/hand.py`: rigged hands with real fingers (palm + 5 fingers x 3 joints, 3D, z-buffered capsules, ink at depth jumps);
  they hold the drumsticks (fulcrum grip, fingers loosen on the backstroke), the mic, the guitar neck and a pick.
- `puppet/motion.py`: HeadMotion (holds, eased moves, accent nods, blinks), Spring, decay_since.
- `puppet/cutout.py`: photo -> sprite helpers (paper flatten, paper cutout, ink-only sprites).
- `puppet/props.py`: mic, drum kit (sparkle shells, chrome hoops, stands; Ben's skull on the kick), guitar, strap, drawn in code.
- `puppet/cast/`: Singer (`distor.py`, Ben's drawing), Drummer (`pepper.py`, David Revoy's Pepper model sheet, CC BY 4.0),
  Guitarist (`coriander.py`, Revoy's Coriander concept art, CC BY 4.0).
- `puppet/song.py`, `puppet/production.py`: Song (score + audio + vocal drivers), stage/camera/shot list/rendering.
  - `python3 clips/silence/clip.py video` -> `out/silence.mp4` (44 s; two halves in parallel, ~8 min)
  - `python3 clips/silence/clip.py stills 3 15 21` / `... singer 8 6` (one character alone)
  - new clip: copy `clips/_template/`
- `puppet/ben_mouths.py`: cuts Ben's mouth poses from his photographed sheet (`drawings/ben/mouths-sheet.jpg`); the singer sings with them.
- `puppet/standins.py`: Distor's stand-in arms (from Ben's new page `drawings/ben/distor-front-v2.png`) and the jacket continued past the paper edge -> `public/parts/distor/`.
- `puppet/cutout_paint.py`: cut-outs of the Revoy paintings (sources in `drawings/revoy/`) and Pepper's arms -> `public/parts/pepper/`.

## Band video (older Remotion version, superseded)

`music/score.py` writes a short pop-rock song as a score and renders it with FluidSynth;
the same score drives the band in `src/band/` (singer, guitarist, drummer, stage, cameras).
`npx remotion render Band out/band.mp4`.
Current test song: `music/song2.py` — an excerpt of "silence in the air" by JeffSpeed68 (Stefan Kartenberg), ccMixter, CC BY-NC 3.0 (https://ccmixter.org/files/JeffSpeed68/53352), plus our drums; mouth shapes from Rhubarb Lip Sync on the vocal stem. Design notes: `docs/DESIGN.md`; what to draw: `docs/BEN_DRAWING_KIT.md`.

## Pipeline

1. **Scan** a drawing into `drawings/` (closed black outlines, plain white paper).
2. **Cut out**: add an entry to `drawings/cutouts.json`, then run
   `python3 scripts/cutout.py --all` → transparent PNGs in `public/chars/`.
3. **Animate** in [Remotion](https://www.remotion.dev) (React): shots live in `src/shots/`,
   reusable parts in `src/lib/` (character layer, tears, line boil, grain, vignette)
   and backgrounds in `src/backgrounds/`.
4. **Preview / render**:
   - `npm install` (once)
   - `npm run studio` — live preview in the browser
   - `npx remotion render Test2c out/test2c.mp4` (also Test1, Test2a, Test2b)

## Layout

```
drawings/         scans + cutouts.json (crop, outline settings, anchor points)
scripts/cutout.py scan -> transparent PNG
public/chars/     cut-out characters
scripts/split_front.py  front drawing -> head / body layers + eye mask (puppet)
public/tex/       texture patches cut from Ben's drawings (skin, hair, jacket)
src/chars/        DistorFront (puppet from Ben's drawing), DistorSide (drawn in code)
src/lib/          Character, Tear, Balloon, Look, anim (blink/talk/sway), effects
src/backgrounds/  GreyCity (drawn in code)
src/shots/        one file per shot
```

## Drawing tips for animation

- One character per page, nothing else on it; sign on the back.
- Close all outlines (no gaps), so the cutout does not leak.
- For moving parts (eyes open/closed, 3–4 mouths, arms), draw each part on its own page.
