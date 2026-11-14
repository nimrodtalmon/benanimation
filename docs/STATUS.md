# Status and next steps (handoff, Nov 14 2026)

Read this first in a new session, then `docs/TOOLKIT.md` (pipeline + lessons).

## Clips
- `clips/silence` (clip 1): the band to "silence in the air" — Distor sings, Pepper drums, Coriander on guitar.
- `clips/soundcheck` (clip 2): Distor walks in, takes the mic.
- `clips/arrival` (clip 3): Distor and Pepper walk in, a cut hides her sitting down, hi-hat count-in, the band.
- `clips/motiontest`: walk + one sung line, options A-E (motion study).
- `clips/band2` (clip 4): clip 1's story in the chosen style (C + D + E, a bit more camera: drift 0.065, crash kicks 0.035, roll 0.5).
- `clips/distorted` (clip 5): Ben's band Distorted Reality plays their own song (instrumental, recorded by the band; `public/songs/dr1`).
  Distor on drums (`cast/distor_drums.py`), the blonde on guitar and the boy from the band page on bass (`cast/bust_player.py`).
- `clips/stylereel`: the same 10 s of the band (song 18-28 s) in many styles; round 1 = styles 1-7, round 2 = A-E (11-15).

## What Nimrod liked / decided
- Motion: fluid mode (`Singer(fluid=True)`), the 3/4 walk, calm heads; the line boil only a little (HandDrawn default boil 0.4).
- From round 1 of the style reel: the TV-static look (style 6: night sky, light from the ground, `TVStatic`) and the camera moves.
- He wants faces with expressions for the drummer and guitarist (`face=True`, `motion.Face`): round 2 tests them.
- Round 2 pick (Nov 14): liked C (colored stage lights), D (worn-tape TV: color bleed, scan lines) and E (one cut per bar),
  with a bit more camera movement than E. So the chosen style = colored stage + `TVStatic(0.14, vhs=0.8)` + faces +
  Clint Eastwood camera (drift ~0.06-0.07, parallax 0.55, ghost 0.35) + bar cuts (`BAR_CUTS` style shot list).

## New toolkit pieces (Nov 14)
- `puppet/transcribe.py`: a band recording + Demucs stems -> score (drum hits per band, bass/guitar notes, chords, beats, sections).
  Check by eye with `tools/score_plot.py`.
- `puppet/photo.py`: phone photo of a page -> even light (paper-only background estimate), figure cut by flood fill.
- `puppet/build_bust.py`: a bust drawing -> puppet body (jacket/hoodie continued past the page, glare removed, stand-in arms).
- `cast/pepper.py` Drummer has a `rig()` hook: any character can sit at the kit (kit units around the shoulders).

## Generic structure (Nov 14, round 5)
- A character = drawings + one settings file `public/rigs/<name>.json` (cut, colors, shoulders, arm/hand sizes,
  head landmarks, how it plays: instrument, strum point, neck angle, strap). `build_bust.py <name>` -> `public/parts/<name>/`.
- `puppet/rig.py` loads any rigged character the same way (puppet config, arms, hand style).
- Cast classes have no character-specific code: `BustPlayer(song, name)` (guitar/bass from the rig's "play"),
  `KitPlayer(song, name)` (any character at the drum kit; `DistorDrummer` = `KitPlayer(song, "distor")`).
- Drums: stick tips land exactly (`tools`-style check: tip vs strike point < 1.5 px): arms unfold when a target is out of
  reach (`Limb.solve_reach`), the stick angles toward a target the wrist cannot reach, a free hand never crosses the kit
  to a far cymbal; strike points per piece in the piece's own radii (`Drummer.STRIKE`: crash on its near edge, hi-hat on
  its open side, drums near the middle); the hi-hat stands above the snare.
- `Production(marker=...)`: how hand-drawn code props look (1 = marker style, 0 = closer to real); the kit got lathed
  cymbals with a lit edge and a shaded bell, smoother coated heads, glossier shells.

## Making a new clip (Nov 14, round 6): song + band -> show
1. `python3 puppet/new_song.py <audio> <id> --credit "..."` -> `public/songs/<id>/` (score with drums, bass, guitar,
   vocal syllables + vowels, beats, breaks; the voice as its own track). Check with `tools/score_plot.py`.
2. A new character: drawings + `public/rigs/<name>.json`, then `python3 puppet/build_bust.py <name>` (kinds: bust with
   stand-in arms, "hoodie" sketch, "portrait" = head and shoulders only, for side views and close-ups).
3. A clip script names only the song, the band and the end card: `Show(Song.load(id), [Member(role, rig, sings=, side=)], ...)`
   (`puppet/show.py`): cast in every shot size (framed from each head's size), shot list, light cues, breaks, flashes,
   titles, tape trick, flash cuts, camera ladder, lip sync and a boom mic for the singer, profile shots.
   Examples: `clips/distorted`, `clips/silence_band`.
- Rendering ~0.58 s/frame (was 0.91): lines drawn only in their box, fast channel means, 2x supersampled flat hands.

## Feedback (Nov 14)
- band2: TV effect too much (keep as a dial: `TVStatic(amount, vhs)`); drummer hand looked detached -> fixed (hand sits in the cuff).
- Ben's band: Distor = drummer, blonde = guitarist (sings most in songs with vocals; all of them sing), bassist = left boys of the band page.

## Next steps
1. Done (Nov 14): `clips/band2`, full song in the chosen style; waiting for Nimrod's feedback.
2. Ben's new pages (app: talmonlab.bgu.ac.il/ben/): head and body turned halfway (3/4) for walking; when they arrive, add a 3/4 body to `Singer` for walks.
3. Clint Eastwood ideas not done yet: repeat short bits forward/backward on the beat ("rock and roll"), freeze frames with titles,
   a camera "ladder" over the song (`Production(drift=lambda t: ...)`), smears on drum hits.
4. Expressions on Revoy's painted guitarist are subtle (small face); Pepper's read well.

## Practical
- Render: ~3-6 fps on 2 CPUs; a 10 s excerpt ~4-6 min, the 44 s song ~8-12 min (two halves in parallel).
- `tools/jerk.py` measures motion smoothness; check stills before sending videos.
