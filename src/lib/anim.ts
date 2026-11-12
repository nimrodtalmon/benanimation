/** Small animation helpers (all take the shot's local frame). */

/** Eyelid closure 0..1 for blinks starting at the given frames (6 frames each). */
export const blinkAt = (frame: number, starts: number[]) => {
  for (const s of starts) {
    const t = frame - s;
    if (t >= 0 && t < 6) return [0.5, 1, 1, 0.7, 0.35, 0.1][t];
  }
  return 0;
};

/** Mouth openness 0..1 while talking in [from, to). Irregular, like syllables. */
export const talk = (frame: number, from: number, to: number) => {
  if (frame < from || frame >= to) return 0;
  const t = frame - from;
  const v = Math.abs(Math.sin(t * 0.55) * Math.sin(t * 0.21 + 1.3));
  return Math.floor(t / 2) % 7 === 6 ? 0 : v; // short pauses
};

/** Slow idle sway, for heads and bodies. */
export const sway = (frame: number, amp: number, period = 60, phase = 0) => amp * Math.sin((frame / period) * Math.PI * 2 + phase);
