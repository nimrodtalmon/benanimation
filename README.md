# benanimation

Animated clips from Ben's hand-drawn characters. Everything is code: scans in, MP4 out.

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
   - `npx remotion render Test1 out/test1.mp4`

## Layout

```
drawings/         scans + cutouts.json (crop, outline settings, anchor points)
scripts/cutout.py scan -> transparent PNG
public/chars/     cut-out characters
src/lib/          Character, Tear, effects (Boil, Grain, Vignette)
src/backgrounds/  GreyCity (drawn in code)
src/shots/        one file per shot
```

## Drawing tips for animation

- One character per page, nothing else on it; sign on the back.
- Close all outlines (no gaps), so the cutout does not leak.
- For moving parts (eyes open/closed, 3–4 mouths, arms), draw each part on its own page.
