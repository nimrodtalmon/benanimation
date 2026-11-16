# Making a clip: the guide

Start here in a new session. A clip = **drawings** (Ben's characters) + **a song** + **a venue**
(where it happens) + **a recipe** (`Show`: who plays, shots, moves, close-ups, look). Everything is
Python (`puppet/`); 3D parts (instruments, kit, close-ups, scenery) are built with Blender as a Python
module and cached. A clip script is ~20 lines (see `clips/_template/clip.py`).

## 0. Setup (fresh machine)

```
pip install --break-system-packages bpy opencv-python numpy librosa demucs soundfile
python3 -c "import bpy, cv2"        # Blender 4.x/5.x as a module, Cycles on CPU
```
Renders are slow on 2 CPUs: ~1 s per frame of video (2 halves in parallel), 3D builds minutes to an
hour (cached under `out/`, never committed).

## 1. A new song

```
python3 puppet/new_song.py path/to/song.mp3 <id> --credit "Song by X (license)"
```
Separates stems (Demucs), transcribes drums/bass/guitar/vocals, beats, sections, breaks ("holes") and
the loudest parts ("peaks") into `public/songs/<id>/score.json`. Check it: `python3 tools/score_plot.py <id>`.
Keep the credit (CC BY-NC songs: the credit must stay in the video; `Song.credit` puts it there).

## 2. New characters from Ben's drawings

Each character is one drawing (head + body, front view) plus a JSON rig:

1. Photo/scan into `drawings/`. Ben draws on plain white paper with closed black outlines; ideally
   also separate arms (upper arm, forearm + hand) and a side view (`docs/BEN_DRAWING_KIT.md`; the upload
   app is `ben-app/`).
2. Write `public/rigs/<name>.json` (copy `blonde.json`): the photo, crop, kind (`bust`, `hoodie`,
   `portrait` for a side head), head landmarks (face oval, eyes, **mouth on the drawn lip line**, chin,
   neck), shoulders, arm size, hand length, eye fills, and `play` (instrument, strum point, 3D model).
3. `python3 puppet/build_bust.py <name>` -> `public/parts/<name>/` (body, arms, parts.json).
4. Check landmarks with an overlay before animating (the mouth must open at the lips, not the chin):
   `python3 tools/rig_overlay.py <name>` -> `out/rig_<name>.jpg` (rig coordinates are pixels of the source photo).
5. In a clip: `Member(role, rig, side=<side-view rig>, sings=True/False, seed=n)`.
   Roles: `drums`, `guitar`, `bass` (others work as front players with no instrument).

## 3. Instruments and the kit

- Guitars/basses: `public/instruments/<name>.json` (body outline, colors, scale length, pickups,
  frets) -> `puppet/guitar3d.py` renders it front-on; a rig's `play.model` names it.
- Drum kits: `public/kits/<name>.json` (pieces, positions in cm, camera) -> `puppet/kit3d.py`;
  `Show(kit=<name>)`.
- Both are rendered once per spec (and per drawing style); 2D just places the picture.

## 4. Where it happens: venues

`puppet/venues.py` (pass `Show(..., venue=...)`):

| venue | what | 3D source |
|---|---|---|
| `Club()` | stage, beams, logo sign, haze, white flashes | none (2D) |
| `Campfire(sky=...)` | intro walk through woods at sunset, then a fire at night | `scenery3d.py` (forest.json), `fire3d.py` |
| `Rooftop(sky=...)` | a city rooftop at night; windows follow the music | `city3d.py` (rooftop.json) |

| `Wilds(moods, weather, giant)` | woods, far mountains, open sky; light, sky and weather follow a story score (the hunt) | `scenery3d.py` (wilds.json, `Wilds3D`) |

Skies (`puppet/skies.py`): `twilight`, `stars`, `clouds`, `aurora`, `dawn`, `day`, `storm`, `blood`, `ash`, or a
schedule `[(t, name), ...]`. Weather (`puppet/weather.py`, stateless, any venue): `Lightning`, `Rain`,
`Particles("ash"/"embers")`, `Birds`, `smoke`, `blaze`.

`Wilds` (`puppet/wilds.py`): moods `[(t, "dawn"|"day"|"red"|"storm"|"ash"|"blood"|"night")]` (sky + land tint +
light passes + light on the cast, crossfaded); weather keyframes `rain/ash/embers/smoke/blaze: [(t, amount)]`,
`strikes: [(t, x)]`, `birds: [(t, (x, y), dir)]`, `shots: [t]` (gun flashes). The view comes from the cast name:
`land*` (side-on, scrolling, no one), `walk*` (scrolling, walking busts), `full*` (side-on, still: someone walks
across), `giant*` (low horizon, the venue's `giant` drawn behind the trees), else the clearing.

**A new venue** = a class with `stage(song)` (an object with `frame(t, wide, cue)` giving the empty
picture), `cue(t, shot)`, `backdrop(frame, t, shot)`, `fore(frame, t, shot)`, `closeup_light(t)`;
`travel=True` gives walking shots in the intro. Light that should fall only on the characters:
`keep_bg()` in backdrop, `light_cast(frame, mult)` in fore. For 3D scenery copy `city3d.py`: a JSON
spec, a Blender `build()` that renders passes (linear `.npy`, open sky as alpha), a 2D class that mixes
the passes per frame cheaply (precompute everything in `__init__`).

## 4b. Props and the hunter

- `puppet/prop3d.py`: hand props in 3D from `public/props/<name>.json` (parts: profile/box/cyl/ball/ring, materials,
  anchors in cm), rendered side-on (+ named perspective views) and drawn; `Prop3D(name).draw(frame, at, anchor,
  angle, scale)`, `.point(...)`, `.draw_at(view, ...)`, `.draw_fit(view, a, pa, b, pb)`. `rifle.json` is the first.
- `cast/hunter.py` `Hunter`: Distor's full drawing (Singer: legs, face) holding the rifle: poses schedule
  `[(t, "carry"|"aim"|"god")]`, `shots=[t]` (recoil, muzzle flash, smoke), `sing=[(t0, t1)]` (mouth to the vocals only
  then), `god(t)` (glowing eyes, dark aura). `Singer.draw_arms()` / `look()` are the hooks it overrides.

## 5. The story: the recipe

```python
show = Show(Song.load(id), BAND, name="myclip", script=__file__, logo=ben_logo(),
            venue=Rooftop(), style3d="ben", closeups=5, moves=True,
            end_lines=["music: ...", "drawings: Ben"])
prod = show.production()
```
`Show` (puppet/show.py) plans the whole video from the song: intro (who plays alone gets the shot),
band entrance, one-bar shots in the body, singers more often, side views, freezes on breaks, tape
tricks, flash cuts late, close-ups of hands on instruments (`closeups=n`), camera moves (push-in,
pull-out, pan, handheld in peaks, whip pans; `moves=False` to turn off), light cues from the venue,
titles and an end card. Players move with `Groove` (calm drift, head-banging in loud parts, jumps in
peaks, a lean on the shout after a break) and `Smoothed` expressions.

For a different story, either:
- pass options to `Show` and pick a venue (most clips), or
- override in a subclass: `plan()` (the shot list: `(t0, t1, cast name or tuple, opts)`), `make_cast()`,
  `post()` (titles); shot opts: `zoom=(a, b)`, `focus=(x, y)` or `focus(t)`, `tmap(t)` (freeze/loop),
  `whip=±1`, `look=False`.
- A story clip (not a band show): subclass `Show`, add cast in `make_cast()` and write `plan()` act by act from the
  song's times (see `clips/hunt/clip.py`: a `run(a, b, cycle, n)` helper cuts n-bar shots taking turns).
- For something not band-shaped at all, use `Production` directly (see `clips/legacy/` for examples).

## 6. One look: drawn

`style3d="ben"` draws everything 3D like Ben's drawings: outlines from the geometry in Blender
(`blendkit.drawn`) + flattened colors and pencil strokes in 2D (`puppet/ink.py`). Builders read the
style from the environment (`BENANIM_STYLE`, set by `ink.use`) and cache drawn renders separately
(`name@ben-...`). `style3d=None` keeps photoreal 3D. New styles: add to `ink.STYLES`.

## 7. Render and check

```
python3 clips/<name>/clip.py stills 3 16 24.5 48      # single frames -> out/<name>-still_<t>.jpg
python3 clips/<name>/clip.py segment 10 20 out/x.mp4  # a piece
python3 clips/<name>/clip.py video                    # all -> out/<name>.mp4
ffmpeg -i out/<name>.mp4 -c:v libx264 -crf 26 -vf scale=1280:-2 -c:a aac out/<name>-share.mp4
```
- Look at stills of every kind of shot before a full render (wide, solo, side, drums, close-up, break).
- First use of a close-up, kit, guitar or venue builds it in Blender (minutes; close-ups ~20 s a
  frame, rendered on threes). To build close-ups ahead of the video, instantiate the show and call
  `show._cast[name]().cu.ensure()` for each `cu_*` shot (see `tools/prebuild.py`).
- `video` renders two halves in parallel; a half that dies (out of memory) is re-rendered alone.
- Smoothness check for a character: `tools/jerk.py`; speed: `tools/profile_clip.py <name>`.
- After changing code: `python3 tools/smoke.py` (one frame of each shot kind in each clip).

## 8. Caches and builders (conventions)

- Every Blender builder: a JSON spec in `public/...`, `build(name)` (runs in a subprocess), a 2D
  class that loads the render. Cache dir = `out/<kind>cache/<name>[@style]-<hash(spec, BUILD)>`.
- **Bump the module's `BUILD` number when you change its Blender side** (geometry, materials, render
  passes). Edits to the 2D side keep the renders. After a deliberate key change, `tools/migrate_caches.py`.
- `blendkit.built(d)` falls back to the newest finished render of the same name, so an interrupted
  build or an edit never blocks a clip.
- `out/` is git-ignored; never commit renders. Credits for third-party assets: MakeHuman hands (CC0,
  `public/hands/README.md`); Revoy paintings in legacy clips (CC BY 4.0).

## 9. Code map

```
puppet/
  show.py         Show recipe (plan, close-ups, moves, cues, titles) -> Production
  production.py   Production: stage, shot list, camera (zoom/focus/drift/parallax/whip), looks, rendering
  venues.py       Club, Campfire, Rooftop, Walking (+ Venue base with cast lighting)
  song.py, new_song.py, transcribe.py      songs: audio -> score
  groove.py, motion.py                      how players move (groove, faces, springs)
  cast/           BustPlayer (any rig with an instrument), Portrait (side view), KitPlayer (any rig at a
                  3D kit), CloseUpShot (3D hands on instruments), legacy casts (distor, pepper, coriander)
  rig.py, build_bust.py, photo.py, warp.py, limb.py, hand.py      drawings -> puppets
  props.py, scenery.py, outdoors.py, skies.py                      2D props, logo/sign/haze, drawn scenery, skies
  blendkit.py     Blender helpers (scene, nodes, renders as linear .npy, drawn outlines, tonemap, caches)
  guitar3d.py, kit3d.py, closeup3d.py, fire3d.py, scenery3d.py, city3d.py   3D builders
  ink.py          the drawn style (2D)
clips/<name>/clip.py   one clip each (campfire, rooftop, distorted, silence_band; old ones in clips/legacy/)
public/           rigs, parts, songs, instruments, kits, fires, scenery, hands (specs and small assets)
tools/            score_plot, jerk, mh_hands (MakeHuman hands), migrate_caches, prebuild, smoke, profile_clip
docs/             GUIDE (this), TOOLKIT (pieces + lessons), STATUS (state, open issues, history of rounds)
```
