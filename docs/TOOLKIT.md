# Animating Ben's drawings: the toolkit

The aim is reusable code for turning Ben's marker drawings into good animation, not one clip.
`clips/silence/clip.py` is one production built with it. This file says what each piece does, how to
add a new character, and what we learned.

## Layout

```
puppet/            the toolkit (generic, reused by every clip)
  cutout.py        photo -> sprite             warp.py     drawing + landmarks -> face/head motion
  ben_mouths.py    mouth sheets                limb.py     bending arms (IK)
  standins.py      code-drawn missing parts    hand.py     rigged hands (+ StickHand)
  props.py         instruments, marker()       motion.py   HeadMotion, Spring, decay_since
  song.py          Song: score + audio + vocal drivers
  production.py    stage, crowd, camera finish, Production (shot list, parallel render, stills)
  cast/            one module per character: distor.py, pepper.py, coriander.py
clips/<name>/clip.py   a clip = Song + cast + SHOTS (see clips/_template/clip.py)
```

A character class has `__init__(song, scale, cx, top)`, `draw(frame, t)`, `j["credit"]`, and
optionally `advance(f)` (springs that must keep running while it is off camera). It reads all
timing from `song` (never from a global), so it works in any clip with any song.

## Pipeline

1. **Photo → sprite.** `puppet/cutout.py`
   - `paper_flatten`: evens out a phone photo.
   - `paper_cutout`: removes a flat paper color around a colored drawing.
   - `ink_sprite`: turns line art (closed eyes, folds) into an alpha sprite in Ben's marker black.
   - Mouths with an outline: `puppet/ben_mouths.py`. You trace the outline once by hand; automatic
     outline detection failed on gaps in the ink.
2. **Character = one drawing + landmarks.** `puppet/warp.py` (`CHARS`, `Puppet`). Landmarks:
   face oval, eyes, mouth (x, y, half width, jaw drop), chin, neck band, head ellipses, a skin
   point. The drawing's own pixels are bent per frame:
   - head turn, tilt and nod, with the neck bending smoothly;
   - hair sway;
   - jaw drop with his drawn mouth stretched into the opening;
   - blinks: skin covers the eye, then his closed-eye strokes are drawn on;
   - irises painted into the eye whites.
3. **Missing parts.** `puppet/standins.py` draws parts Ben hasn't drawn yet, in his marker style:
   arms, the jacket past the page edge, shirt, jeans, shoes. When he draws a part, swap the file
   in; the joints stay in `parts.json`.
4. **Bending limbs.** `puppet/limb.py`: upper arm + forearm (two drawings, or one painted arm),
   2-bone IK, a round outer elbow, a crease shadow, and forearm foreshortening (`fore_k`).
   `part="upper"/"fore"` lets you draw other layers in between, e.g. a guitar.
5. **Hands.** `puppet/hand.py`: a 3D hand with 5 fingers × 3 joints, drawn with ink at depth jumps,
   contact shadows and skin grain.
   - Poses: relaxed, fist, stick, mic, fret, pick.
   - Held objects live in the hand's frame, so fingers wrap around them; a `hidden` object occludes
     without being drawn (e.g. the guitar neck).
   - `Style(flat=True)` gives Ben's flat look.
6. **Props.** `puppet/props.py`: mic, drums, cymbals, guitar, strap, stands.
   - `marker()` re-renders any code-drawn sprite in a marker look: flat tones, streaks, wobbly
     ink, white highlight dashes.
7. **Motion.** `puppet/motion.py`
   - `Path` (eased keyframed positions) and `Walk` (front-view walk: planted feet, swing with a
     lift, hip bob). A character given `walk=Path(...)` draws separate legs (`limb.Limb` with
     thigh/shin foreshortening `up_k`/`fore_k`).
   - `Spring.at(target, t)` catches up frame by frame, so a still equals the same video frame.
   - `HeadMotion`: holds a pose, eases to a new one, nods on accents, blinks at irregular times.
   - `Spring`: smooth following motion.
   - `decay_since`: jumps on a hit, then decays.
   - Every move is timed from `src/song.json` (drums, strums, sung notes) and the vocal loudness.
8. **Props in the world.** `props.mic_stand`; `hand.render(..., show_hand=False)` draws a held object
   alone (a mic on a stand) so a character can pick it up (`Singer(stand=dict(x, grab))`).
9. **Shots.** `production.py` (`Production`): stage lights, crowd, camera push, a shot list,
   credits. It renders the clip in two halves in parallel. `stills` renders single frames for checking.
   A shot may carry options `(t0, t1, who, dict(zoom=(a, b), focus=(x, y) or focus(t)))` for push-ins
   and pans; `Production(cue=...)` sets lights per moment (`beams`, a white `spot` at x, `house` level).

## Adding a character (checklist)

1. Ben uploads head, body, upper arm, forearm + hand (and a leg) in the app (`ben-app/`).
2. Cut out each page (`cutout.paper_cutout`). Put the landmarks in a `CHARS` entry. Check them on a
   gridded crop: draw a grid over the image and read the coordinates.
3. Joints in `parts.json`: shoulder and elbow on the upper arm, elbow and wrist (+ grip) on the
   forearm. One arm drawing serves both arms (mirrored).
4. Optional: mouth sheet (`ben_mouths.py`), closed eye (`cutout.ink_sprite`), iris color.
5. Write `puppet/cast/<name>.py` with `draw(frame, t)` that maps song events to targets. Use
   `motion.py` for the timing, IK for the arms, `hand.py` for anything held.
6. Add the character to a clip's CAST. Run `clip.py stills ...` and look at the frames before
   `clip.py video`.

## Lessons so far

- **Bend the drawing's own pixels** (Live2D style) rather than moving rigid parts: it keeps Ben's
  line and texture.
- **Ben's own drawings beat code stand-ins** every time (the mouth sheet, the closed eye). Ask him
  for small extra drawings (expressions, closed eyes, hands) instead of synthesizing them.
- **Motion:**
  - Constant noise and beat-locked bobbing look mechanical.
  - Holds, slow eased moves (spring ω≈4, ζ≈0.9) and a few accents look alive.
  - Big overshoot looks violent.
- **Hands:**
  - Check every grip from all sides on a test sheet (`hand.Pose` × `orient` grid) before using it.
    Flipped fingers come from a wrong roll/pitch or a mirrored hand.
  - Rule for a held object: decide which side of the hand faces the camera first, then put the
    object where the fingers close.
- **Attachment** (guitar on body, mic in hand) needs three things: the joints line up, a contact
  shadow, and correct layer order. The `limb` `part=` split and the hand's `hidden` objects exist
  for this.
- **Painted characters with arms in the drawing:** cut the arm out, inpaint the body behind it,
  and bend only the forearm. Moving the whole painted arm breaks the silhouette.
- **Style consistency:** run code-drawn props through `props.marker()`. Smooth CG shading next to
  marker drawings looks wrong.
- **Check frames** at full size, on crops, for every shot, before each full render.
- **Front-view walking:** a foot's travel along the walk mostly goes toward or away from the camera.
  Show only ~a quarter of it sideways and show the lift as a shorter leg. Full sideways travel reads
  as kicking.
- **Springs must be time-aware.** Otherwise stills (and the first frame of a shot) show arms flung
  around by a spring that skipped time.
- **Clip 2 (`clips/soundcheck`)** was written against the toolkit only. What it needed and added
  there: walking, legs, picking up a prop, light cues, camera pans. The next clip should need
  less new code.
