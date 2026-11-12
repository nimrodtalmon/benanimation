# Design notes (research, Nov 2026)

## What the references do
- Gorillaz "Clint Eastwood" (Passion Pictures, 2001): 2D characters in a CG world; the studio's 2D department was a whole floor. Later work kept flat 2D characters, often on photographic or painted backgrounds, and used a low frame rate for a stop-motion / puppet feel.
- Digital cut-out (South Park, Toon Boom Harmony productions): characters are puppets made of parts in a hierarchy (arm > forearm > hand), each with a pivot; hands, feet and faces are swapped as drawings ("substitution") instead of deformed.

## Levers that make cut-out convincing (in order of payoff)
1. **Rig with real pivots and a hierarchy**, plus 2-bone IK so hands land on drums, strings, mic.
2. **Substitution drawings** for mouths (6 shapes A–F, Hanna-Barbera set used by Rhubarb Lip Sync), eyes, hands.
3. **Timing from the music score**: every hit, strum and sung note is known exactly, so motion has anticipation before a hit and follow-through after.
4. **Overlapping action**: hair, jacket, head lag behind the body (damped springs driven by beats).
5. **Animate on twos** (pose changes every 2nd frame): hand-made, puppet feel. Camera stays smooth.
6. **Staging and light**: one consistent set, floor shadows, cel shading (one shadow tone), colored stage light on the beat, a few camera angles cut on phrases.

## Architecture
- `music/score.py` writes the song as a score (JSON + MIDI); FluidSynth renders the audio. The same score drives the animation.
- Characters are rigs in code (`src/band/`): parts + pivots + poses. Later Ben draws the parts on paper (one part per page) and they replace the code drawings 1:1.
- One stage in world coordinates; shots are cameras on that world.
- For real vocals later: Rhubarb Lip Sync (CLI, Linux) turns a vocal track into mouth-shape timings.

## Not used (and why)
- Blender Grease Pencil from Python: rendering needs a GPU/OpenGL display, which this server lacks.
- AI video generators: free tiers are limited, and characters drift off-model.
