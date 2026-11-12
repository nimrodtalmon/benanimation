import analysis from "../clip-analysis.json";

/**
 * Song analysis (from scripts/analyze_audio.py): beats in seconds and
 * per-frame (30 fps) loudness curves, each scaled to 0..1.
 */
export const FPS = 30;
export const SONG = analysis as {
  duration: number;
  tempo: number;
  beats: number[];
  rms: number[];
  bass: number[];
  mid: number[];
  onset: number[];
};
export const SONG_FRAMES = Math.floor(SONG.duration * FPS);

const at = (arr: number[], f: number) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(f)))] ?? 0;

/** Nearest beat to a time (seconds) -> frame. Used to snap cuts to the music. */
export const snap = (sec: number) => {
  let best = SONG.beats[0];
  for (const b of SONG.beats) if (Math.abs(b - sec) < Math.abs(best - sec)) best = b;
  return Math.round(best * FPS);
};

/** Frames since the last beat, and that beat's index. */
export const lastBeat = (frame: number) => {
  const t = frame / FPS;
  let i = -1;
  for (let k = 0; k < SONG.beats.length && SONG.beats[k] <= t; k++) i = k;
  return { index: i, since: i < 0 ? 999 : Math.round((t - SONG.beats[i]) * FPS) };
};

/** 1 on the beat, decaying to 0 over `len` frames. */
export const beatPulse = (frame: number, len = 8) => {
  const { since } = lastBeat(frame);
  return since < len ? Math.pow(1 - since / len, 2) : 0;
};

/** Mouth openness from the song's mid band (where voice and guitars sit), smoothed and gated. */
export const singing = (globalFrame: number) => {
  const v = (at(SONG.mid, globalFrame) + at(SONG.mid, globalFrame - 1) + at(SONG.mid, globalFrame - 2)) / 3;
  const onset = at(SONG.onset, globalFrame);
  const m = Math.max(0, (v - 0.35) / 0.65) * (0.55 + 0.45 * onset);
  return Math.min(1, m * 1.3);
};

export const energy = (f: number) => at(SONG.rms, f);
export const onsetAt = (f: number) => at(SONG.onset, f);
