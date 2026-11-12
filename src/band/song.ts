import song from "../song.json";

/** The generated song's score (music/score.py), in seconds. Drives all band motion. */
export type DrumHit = { t: number; piece: number; vel: number };
export type Strum = { t: number; dur: number; root: number; chord: string; dir: "down" | "up"; accent: boolean };
export type Note = { t: number; dur: number; pitch: number; vowel: string };
export type Section = { name: string; t0: number; t1: number };

export const SONG = song as unknown as {
  bpm: number;
  duration: number;
  drums: DrumHit[];
  guitar: Strum[];
  bass: { t: number; dur: number; note: number }[];
  vocal: Note[];
  sections: Section[];
};
export const FPS = 30;
export const SPB = 60 / SONG.bpm;
export const DRUM = { KICK: 36, SNARE: 38, HAT: 42, OPEN_HAT: 46, CRASH: 49, TOM_HI: 50, TOM_MID: 47, TOM_LO: 45 };

/** Time used for character poses: "on twos" (a new pose every 2nd frame) for a hand-made feel. */
export const onTwos = (frame: number) => (frame - (frame % 2)) / FPS;

export const sectionAt = (t: number) => SONG.sections.find((s) => t >= s.t0 && t < s.t1)?.name ?? "outro";

/** Damped ring after each event time: sum of decaying sines (hair, cymbals, body follow-through). */
export const ring = (t: number, times: number[], tau = 0.25, hz = 3, amp = 1) => {
  let v = 0;
  for (let i = times.length - 1; i >= 0; i--) {
    const dt = t - times[i];
    if (dt < 0) continue;
    if (dt > tau * 6) break;
    v += Math.exp(-dt / tau) * Math.sin(2 * Math.PI * hz * dt);
  }
  return v * amp;
};

/** 1 at the event, decaying to 0 (for flashes, squash). */
export const decay = (t: number, times: number[], tau = 0.12) => {
  let v = 0;
  for (let i = times.length - 1; i >= 0; i--) {
    const dt = t - times[i];
    if (dt < 0) continue;
    if (dt > tau * 6) break;
    v = Math.max(v, Math.exp(-dt / tau));
  }
  return v;
};

export const beatTimes = Array.from({ length: Math.ceil(SONG.duration / SPB) }, (_, i) => i * SPB);
export const barTimes = beatTimes.filter((_, i) => i % 4 === 0);

/** Last event before t and the next one. */
export const around = <T extends { t: number }>(list: T[], t: number) => {
  let i = -1;
  for (let k = 0; k < list.length && list[k].t <= t; k++) i = k;
  return { prev: i >= 0 ? list[i] : undefined, next: list[i + 1], index: i };
};
