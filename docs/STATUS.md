# Status and next steps (handoff, Nov 14 2026)

Read this first in a new session, then `docs/TOOLKIT.md` (pipeline + lessons).

## Clips
- `clips/silence` (clip 1): the band to "silence in the air" — Distor sings, Pepper drums, Coriander on guitar.
- `clips/soundcheck` (clip 2): Distor walks in, takes the mic.
- `clips/arrival` (clip 3): Distor and Pepper walk in, a cut hides her sitting down, hi-hat count-in, the band.
- `clips/motiontest`: walk + one sung line, options A-E (motion study).
- `clips/band2` (clip 4): clip 1's story in the chosen style (C + D + E, a bit more camera: drift 0.065, crash kicks 0.035, roll 0.5).
- `clips/stylereel`: the same 10 s of the band (song 18-28 s) in many styles; round 1 = styles 1-7, round 2 = A-E (11-15).

## What Nimrod liked / decided
- Motion: fluid mode (`Singer(fluid=True)`), the 3/4 walk, calm heads; the line boil only a little (HandDrawn default boil 0.4).
- From round 1 of the style reel: the TV-static look (style 6: night sky, light from the ground, `TVStatic`) and the camera moves.
- He wants faces with expressions for the drummer and guitarist (`face=True`, `motion.Face`): round 2 tests them.
- Round 2 pick (Nov 14): liked C (colored stage lights), D (worn-tape TV: color bleed, scan lines) and E (one cut per bar),
  with a bit more camera movement than E. So the chosen style = colored stage + `TVStatic(0.14, vhs=0.8)` + faces +
  Clint Eastwood camera (drift ~0.06-0.07, parallax 0.55, ghost 0.35) + bar cuts (`BAR_CUTS` style shot list).

## Next steps
1. Done (Nov 14): `clips/band2`, full song in the chosen style; waiting for Nimrod's feedback.
2. Ben's new pages (app: talmonlab.bgu.ac.il/ben/): head and body turned halfway (3/4) for walking; when they arrive, add a 3/4 body to `Singer` for walks.
3. Clint Eastwood ideas not done yet: repeat short bits forward/backward on the beat ("rock and roll"), freeze frames with titles,
   a camera "ladder" over the song (`Production(drift=lambda t: ...)`), smears on drum hits.
4. Expressions on Revoy's painted guitarist are subtle (small face); Pepper's read well.

## Practical
- Render: ~3-6 fps on 2 CPUs; a 10 s excerpt ~4-6 min, the 44 s song ~8-12 min (two halves in parallel).
- `tools/jerk.py` measures motion smoothness; check stills before sending videos.
