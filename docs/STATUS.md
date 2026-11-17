# Status (handoff, Nov 16 2026, round 14)

New session: read `docs/GUIDE.md` (how to make a clip), then this file; `docs/TOOLKIT.md` has the pieces and lessons.

## Now
- Current clips (all `Show`-based, song dr1 by the band Distorted Reality): `clips/distorted` (club), `clips/campfire`
  (walk in the woods, then a 3D fire at night), `clips/rooftop` (city rooftop at night, drawn 3D look; Nimrod: "very
  nice"). `clips/silence_band`: the same band to "silence in the air" (CC BY-NC; keep the credit).
- The look he chose: 2.5D drawn (`style3d="ben"`): Ben's drawings as puppets, every 3D part outlined and flattened.
- Speed (2 CPUs): ~0.7 s per frame for band shots, ~0.4 s for cached close-ups (`tools/profile_clip.py`); a 3-min
  clip ~1 h with two halves in parallel. New close-ups ~20 s a frame in Blender, rendered on threes.
- Checks: `tools/smoke.py` (one frame of each shot kind per clip), `ruff check --select F,E9 puppet tools clips`.
- Old clips (Production API with Revoy's Pepper/Coriander, motion and style studies) are in `clips/legacy/`; the
  first Remotion version in `legacy/remotion/` (not maintained).

- New (Nov 16, round 14): `clips/hunt`, a story clip (not a band show) to a song Nimrod gave (public/songs/hunt;
  the audio is NOT in the repo, which is public: only score.json; a new session needs the mp3 again and
  `python3 puppet/new_song.py <mp3> hunt --credit "..."`). Distor is a hunter with a 3D rifle; acts: the eye, nature,
  the hunt, the storm (lightning hits him), ash (he rises, glowing eyes), the god (a giant behind the burning trees).
  New pieces: `puppet/wilds.py` (venue: moods + weather scores), `puppet/weather.py`, skies dawn/day/storm/blood/ash,
  `scenery3d` wilds.json (far mountains layer, clearing sun + blaze passes, `Wilds3D` drawn in ink),
  `puppet/prop3d.py` + `public/props/rifle.json`, `cast/hunter.py` (Singer got `draw_arms()` / `look()` hooks).
  State: stills and a 14 s segment checked; full render waiting for Nimrod's OK; song credit still unknown.
  Note: scenery3d "far" was rebuilt alone after a fix (camera clip_end) without a BUILD bump.

- New (Nov 17, round 15): `clips/achzar`, Nimrod's own song (Hoodska, "Tsayad Achzar"; his rights; the mp3 is kept
  out of the public repo until he says otherwise), the hunt story with six captioned experiments: classic cuts, long
  takes, comic panels, beat cuts, noir duotone on twos, silhouettes on worn tape. Rendered (out/achzar.mp4, ~1.5 h on
  2 cores); waiting for his picks. Slow spot: the extreme close-up (hunter_eye, scale 1.9) takes ~12 s a frame.
  Round 2 (his feedback: likes the comics; hand on rifle and walking were off; more camera variety; a better story):
  new story (scope, the bassist shot and back, the hunter hunted by the storm, the god crumbles to ash, dawn, the
  book closes); comic lettering and pages throughout; rifle: the rifle follows a smoothed pose and both fists sit on
  it, arms foreshortened with elbows out; walking: front-facing walks are now toward the camera (no sideways
  crab-walk with crossing legs).
  Fix: Production now tells the venue the background after the parallax warp (`keep_bg`), so `light_cast` /
  `cast_mask` see only the cast (before, the warp made every pixel look like cast: whole-frame tints).

## Open issues / ideas
1. Ben's 3/4 and side drawings for walks (upload app talmonlab.bgu.ac.il/ben/); front-only rigs walk stiffly.
2. Close-up hands are generic (MakeHuman), colored from the rig; no close-ups of singers' faces yet.
3. Venues are hand-written classes; a new place = a new builder (copy `city3d.py`) + a Venue class (GUIDE §4).
4. Memory: ~2.6 GB per render process; two in parallel fit, more do not.

---

# History of the rounds (Nov 2026)

## Clips (rounds 1-5)
- `clips/legacy/silence` (clip 1): the band to "silence in the air" — Distor sings, Pepper drums, Coriander on guitar.
- `clips/legacy/soundcheck` (clip 2): Distor walks in, takes the mic.
- `clips/legacy/arrival` (clip 3): Distor and Pepper walk in, a cut hides her sitting down, hi-hat count-in, the band.
- `clips/legacy/motiontest`: walk + one sung line, options A-E (motion study).
- `clips/legacy/band2` (clip 4): clip 1's story in the chosen style (C + D + E, a bit more camera: drift 0.065, crash kicks 0.035, roll 0.5).
- `clips/distorted` (clip 5): Ben's band Distorted Reality plays their own song (instrumental, recorded by the band; `public/songs/dr1`).
  Distor on drums (`cast/distor_drums.py`), the blonde on guitar and the boy from the band page on bass (`cast/bust_player.py`).
- `clips/legacy/stylereel`: the same 10 s of the band (song 18-28 s) in many styles; round 1 = styles 1-7, round 2 = A-E (11-15).

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

## 3D drum kit (Nov 14, round 7)
- `puppet/kit3d.py`: the kit is built in Blender (`pip install bpy`, Cycles on the CPU) from `public/kits/<name>.json`
  (pieces with sizes and places in cm, finishes, camera), rendered once (~5 min) into `out/kitcache/<name>-<hash>/`:
  a static layer (with floor shadows) and states of the moving cymbals (crash wobble angles, hi-hat heights), each cut
  where something is in front of it. meta.json maps the render to kit units and gives strike points per hand from 3D.
- `Drummer(kit="classic", kit_size=1.15)` / `KitPlayer` / `Show(kit=...)` use it; `kit=None` = the old code-drawn kit.
  kit_size makes the kit bigger next to big-headed cartoon players without re-rendering.
- Layout lessons: the snare must sit near the middle (the near hand must reach it) and not behind a rack tom (a hit
  there reads as a tom hit) -> a 4-piece rock kit (one rack tom). Pieces a kit lacks fall back to the nearest one.

## Round 8 (Nov 15)
- Kit: crash and ride moved toward the camera; cymbals listed in the kit's "front" are drawn over the drummer's arms
  (he hits the crash's far edge from behind).
- `puppet/guitar3d.py`: guitars and basses rendered in Blender from `public/instruments/<name>.json` (outline, scale
  length, pickups, knobs, finish); strings stay 2D (they ring). A rig's "play" names its "model"; no model = the old
  drawn guitar. lp_cherry (single-cut, cherry sunburst), pbass_blue (P-bass style, blue burst, maple neck).
- Faces: `motion.Smoothed` = an expression curve over the whole song, Gaussian-smoothed per key (eases in and out,
  anticipates events); no squint per strum. Used by BustPlayer, Portrait, Drummer.
- Peaks: `transcribe.loud_parts` finds the song's biggest sections (loudness plateaus within 3% of the loudest, at most half
  the song) -> score "peaks"; Groove style "mix" goes all-out there: head-banging on every beat (snap down, tip
  forward, slower up; harder on bar starts), hard body rocking, bar-start dips; eased in/out at the edges.

## Round 9 (Nov 15): venues, a second story on the same song
- Fixes: rack tom moved toward the camera (`public/kits/classic.json`); mouth lines moved up to the drawn lips and chins
  set at the real chin (blonde, bassboy, blonde_side) so the jaw opens at the mouth; guitarist arms/hands thicker.
  Lesson: put a rig's "mouth" y on the drawn lip line (check with an overlay), not between lips and chin.
- `puppet/venues.py`: where a Show happens is a plug-in. A venue gives the stage (`frame(t, wide, cue)`), light cues,
  backdrop and fore. `Club()` = the old stage (spotlight, beams, sign, haze, white flashes). `Campfire()` = the intro is
  a walk through woods at sunset (venue.travel -> Show adds walking busts `walk_<role>`, `walk_<role>_close`, the pair
  `walk_w_<role>`; the drummer waits at the destination), then a clearing at night with a fire that follows the band
  (heat curve: embers in breaks, flare-ups on shouts, more in peaks, dies after the last hit).
- `puppet/outdoors.py`: code-drawn scenery: `Sky` (sunset -> night, sun, stars, crescent moon, far hills, pine
  ridge, near pines, clearing edge and ground, ferns in front; layers wrap and scroll at their own speeds),
  `firelight` (ambient dusk/moonlight x picture + flickering warm falloff from the fire, also off-screen),
  `Fire` (stones, logs, embers, banded cartoon flames, glow), `Sparks` (stateless particles: stream with heat, bursts).
- `Walking(player, song)` wraps any bust player (one step a beat). `Production(stage=<object>)` accepts any stage.
- New clip: `clips/campfire` = clips/distorted with `venue=Campfire()`.

## Round 10 (Nov 15): real fire and woods, instrument close-ups
- Feedback: the drawn fire looked like 90s clip art -> real 3D everywhere (like the drums); add close-ups of hands on
  instruments (realistic, detailed; Gorillaz-style inserts).
- `puppet/blendkit.py`: shared Blender helpers (scene, node building, lights, cameras, renders saved as linear .npy,
  `tonemap` (hue-keeping), `built()` = newest finished cache when a builder was edited after rendering).
- `puppet/fire3d.py` + `public/fires/campfire.json`: logs (bark, charred middle, glowing cracks), stone ring, ash bed
  with coals, a volumetric flame (rising 4D noise in a tapered column, fire color ramp) as a seamless loop. Passes:
  moonlit, firelit, embers (linear, summed per frame with the heat), flame (screened, height follows the heat).
- `puppet/scenery3d.py` + `public/scenery/forest.json`: procedural spruces (trunk + whorls of branch cards and twigs
  with needle alpha, light through the needles), ferns, grass (hair), mountains, physical sky (Nishita). Walk layers
  rendered side-on, one period wide, wrapping (parallax scroll); sky at sunset and dusk; the night clearing as two
  passes (moon, fire) relit per frame. Stars and moon added in 2D over the sky.
- `Campfire(real=True)` uses them; firelight now falls only on the cast (found by diffing against the backdrop).
  `real=False` keeps the drawn version.
- `puppet/closeup3d.py`: close-ups of hands on any 3D instrument from the score: notes -> (string, fret) with a
  tuning lowered to fit the song; the fretting hand slides between positions, one finger a fret, thumb behind the
  neck; the plucking hand alternates index/middle rest strokes; strings shake. Hands: skin-modifier skeleton
  (knuckle bulges, nails, subsurface skin in the rig's colors, the rig's sleeve). Light presets per venue
  (sunset / fire / stage). `Show(closeups=4)` cuts them in (intro: fret then pluck; the rest spread evenly).
  Rendered on threes; ~25-50 s a frame on 2 CPUs.

## Round 11 (Nov 15): less clay, skies, drum and pick close-ups
- Feedback: close-up hands looked like clay; try different skies; add drum and pick close-ups.
- Hands (`puppet/closeup3d.py` Hand3D): one mesh per hand made once at rest (skin modifier, subdivided), weights from
  Blender's bone heat on an armature of the palm/finger/thumb bones, bent per frame by dual-quaternion skinning (no
  collapse at bent knuckles). Skin detail is tied to the rest surface (stored rest position, joint distance, back/palm
  side, tip distance): knuckle wrinkles, palm creases, pores, redness at joints and tips, SSS, glossy nails with pale
  free edges. Close-ups skip the paper/TV look (`opts look=False`), render at 1280x720.
- Kinds: fret, pluck (bass), pick (guitar: the hand is placed so the pick sits on the strings, sweeps down/up with the
  strums), sticks (drums: kit3d.model() builds the kit in the shot; matched grip, sticks rise and fall to the score's
  snare/hat hits, the hat shivers; camera over the drummer's shoulder). `kit3d.model()` and `guitar3d.model()` are
  now reusable scene builders. Builders fall back to their newest finished cache after edits (`blendkit.built`).
- `Show(closeups=5)`: intro (bass fret then pluck), then drums, guitar pick, bass, drums spread over the song.
- Skies (`puppet/skies.py`): twilight, stars (Milky Way, airglow, twinkling), clouds (full moon, drifting moonlit
  clouds), aurora (moving curtains). The clearing plate now has an open (transparent) sky; `Campfire(sky=...)` takes a
  name, a schedule, or "tour" (twilight -> stars -> aurora in the first peak -> clouds -> stars).

## Round 12 (Nov 15): real hand anatomy
- Feedback: hands still looked like clay (tubes around a stick skeleton).
- Hands now come from the MakeHuman base mesh and default rig (CC0): `tools/mh_hands.py` cuts both hands (+ some
  forearm) with their finger bones and skin weights into `public/hands/mh_<side>.npz` (hand frame, meters). Hand3D
  bends them by dual-quaternion skinning from our IK chains (each hand's own knuckles and bone lengths), adds a
  subdivision, keeps the skin shader detail and nails (sized from the mesh's fingertips). `pose(reach=True)` moves the
  hand until the fingertips get to their targets (works for any hand size).

## Round 13 (Nov 16): one drawn look, a rooftop, more movement
- Feedback: pick/pluck shots; close-ups still clay-ish; the clip is inconsistent (drawn long shots vs 3D close-ups vs
  a doll-like drum POV) -> one look, preferably closer to the drawings; another background; more movement; keep the
  code modular for later.
- Decision: 2.5D drawn. The characters stay Ben's drawings; everything 3D is drawn too: Freestyle outlines from the
  geometry (`blendkit.drawn()`), then 2D color flattening (`puppet/ink.py`: edge-preserving smoothing, soft lightness
  bands, lighter/paler, pencil strokes in the darker bands; `ink.draw()` adds picture-based lines for sources without
  geometry lines). No depth of field in drawn mode. `Show(style3d="ben")` -> `ink.use()` -> env BENANIM_STYLE ->
  builders (kit3d, guitar3d, closeup3d, city3d) draw outlines and cache separately (`name@ben-hash`); loaders flatten
  colors through `ink.flat_cached()` (disk cache out/inkcache).
- Close-ups: a real pick grip (index under the pick, thumb pressing from the back, the hand placed so the pick meets
  the strings); plucking shot reframed; light preset "rooftop".
- `puppet/city3d.py` + `public/scenery/rooftop.json`: a city rooftop (tar roof with puddles, parapet, water tower,
  stair shed with a lamp, AC units, antenna with a red light, strings of bulbs) and a few hundred buildings with
  random lit windows. Passes: base, windows' light, window ids -> `City3D.frame(windows=share, blackout=...)`.
- `venues.Rooftop`: windows come on in bursts on the beat through the intro, blackout in breaks, back on the shout,
  pulse with the kick in peaks; warm/cool light on the cast only (`Venue.light_cast`, now shared by all venues).
- Movement: `Groove.jump()` (hops on alternating bars in peaks, players alternate by seed), `Groove.lean()` (lean on
  the shout after a break); `Show.add_moves()` (push-in / pull-out / pan in turns, handheld shake in peaks, whip pans
  into the entrance, each peak, each shout; `production.whip()`).
- Rendering: `render_all` re-renders a part that died (out of memory) alone and stitches; scenery layers in half
  floats, flames in uint8.
- New clip: `clips/rooftop`.

## Feedback (Nov 14)
- band2: TV effect too much (keep as a dial: `TVStatic(amount, vhs)`); drummer hand looked detached -> fixed (hand sits in the cuff).
- Ben's band: Distor = drummer, blonde = guitarist (sings most in songs with vocals; all of them sing), bassist = left boys of the band page.

## Ideas from Nov 14 (some done since in Show: freezes, tape tricks, camera ladder)
- Repeat short bits forward/backward on the beat ("rock and roll"), smears on drum hits.
