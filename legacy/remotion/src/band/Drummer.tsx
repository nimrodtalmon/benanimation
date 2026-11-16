import { staticFile } from "remotion";
import { DRUM, SONG, SPB, decay, ring } from "./song";
import { At, FloorShadow, Hand, INK, Seg, V, add, aim, ikArm, ink, len, local, mul, sub } from "./rig";

/**
 * The drummer behind the kit. Each stick follows the drum hits in the score:
 * lift (anticipation), strike, rebound. Cymbals and the kick head react to hits.
 */

const SKIN = "#c98f66";
const SKIN_SH = "#a8704c";
const SHIRT = "#3f6b4a";
const SHIRT_SH = "#2b4c33";

// stick-tip targets on the kit (world)
const POS: Record<string, V> = {
  hat: [752, 676],
  snare: [842, 722],
  tomHi: [906, 642],
  tomMid: [1018, 642],
  floor: [1122, 752],
  crash: [1150, 556],
  rest: [880, 690],
};
const pieceTarget = (p: number) =>
  p === DRUM.HAT || p === DRUM.OPEN_HAT ? "hat" : p === DRUM.SNARE ? "snare" : p === DRUM.TOM_HI ? "tomHi" : p === DRUM.TOM_MID ? "tomMid" : p === DRUM.TOM_LO ? "floor" : p === DRUM.CRASH ? "crash" : "";

type Hit = { t: number; at: string };
/** Split the score's hits between the two hands (left: hi-hat; right: the rest; fast runs alternate). */
const HANDS: { L: Hit[]; R: Hit[] } = (() => {
  const L: Hit[] = [];
  const R: Hit[] = [];
  for (const d of SONG.drums) {
    const at = pieceTarget(d.piece);
    if (!at) continue;
    if (at === "hat") {
      if (!L.length || d.t - L[L.length - 1].t > 0.05) L.push({ t: d.t, at });
      continue;
    }
    const lastR = R[R.length - 1];
    const lastL = L[L.length - 1];
    if (lastR && d.t - lastR.t < 0.2 && (!lastL || d.t - lastL.t > 0.12)) L.push({ t: d.t, at });
    else R.push({ t: d.t, at });
  }
  L.sort((a, b) => a.t - b.t);
  return { L, R };
})();

const smooth = (u: number) => {
  const x = Math.min(1, Math.max(0, u));
  return x * x * (3 - 2 * x);
};

/** Stick tip and lift for one hand at time t. */
const stick = (hits: Hit[], t: number, shoulder: V) => {
  let i = -1;
  for (let k = 0; k < hits.length && hits[k].t <= t; k++) i = k;
  const prev = hits[i];
  const next = hits[i + 1];
  const A = POS[prev?.at ?? "rest"];
  const Bp = POS[next?.at ?? prev?.at ?? "rest"];
  const t0 = prev?.t ?? -1;
  const t1 = next?.t ?? t0 + 10;
  const D = t1 - t0;
  const x = t - t0; // since last hit
  const y = t1 - t; // until next hit
  const peak = Math.min(75, 28 + 90 * Math.min(D, 0.6));
  const up = (v: number, k: number) => (v < k ? peak * (v / k) : peak);
  let lift = Math.min(up(x, 0.08), up(y, 0.07));
  if (x > 0.3 && y > 0.4) lift = Math.min(lift, 34);
  if (!prev) lift = 34;
  const tip = add(lerpV(A, Bp, smooth(D > 0 ? (x - 0.05) / Math.max(0.05, D - 0.12) : 0)), [0, -lift * 0.6]);
  // hand sits back along the line to the shoulder; the stick tilts up as it lifts
  const toShoulder = sub(shoulder, tip);
  const hand = add(tip, add(mul(toShoulder, 72 / len(toShoulder)), [0, 8 - lift * 0.35]));
  return { tip, hand, lift };
};
const lerpV = (a: V, b: V, u: number): V => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];

const times = (pred: (p: number) => boolean) => SONG.drums.filter((d) => pred(d.piece)).map((d) => d.t);
const CRASH_T = times((p) => p === DRUM.CRASH);
const KICK_T = times((p) => p === DRUM.KICK);
const SNARE_T = times((p) => p === DRUM.SNARE);
const HAT_T = times((p) => p === DRUM.HAT || p === DRUM.OPEN_HAT);
const TOM_T = (p: number) => times((q) => q === p);

const Drum: React.FC<{ c: V; rx: number; ry: number; h: number; hit: number; color?: string }> = ({ c, rx, ry, h, hit, color = "#8c1c2a" }) => (
  <g>
    <path d={`M ${c[0] - rx} ${c[1]} L ${c[0] - rx} ${c[1] + h} A ${rx} ${ry} 0 0 0 ${c[0] + rx} ${c[1] + h} L ${c[0] + rx} ${c[1]}`} fill={color} {...ink} />
    <path d={`M ${c[0] + rx * 0.4} ${c[1] + ry * 0.9} L ${c[0] + rx * 0.4} ${c[1] + h + ry * 0.9} L ${c[0] + rx} ${c[1] + h} L ${c[0] + rx} ${c[1]} Z`} fill="#000" opacity={0.25} />
    <ellipse cx={c[0]} cy={c[1]} rx={rx} ry={ry * (1 - 0.15 * hit)} fill="#ece6d6" {...ink} />
    <ellipse cx={c[0]} cy={c[1]} rx={rx * 0.86} ry={ry * 0.7} fill="none" stroke="#bdb5a0" strokeWidth={2} />
  </g>
);

const Cymbal: React.FC<{ c: V; rx: number; deg: number; stand?: number }> = ({ c, rx, deg, stand = 0 }) => (
  <g>
    {stand > 0 && <line x1={c[0]} y1={c[1]} x2={c[0]} y2={c[1] + stand} stroke="#555" strokeWidth={6} />}
    <g transform={`translate(${c[0]} ${c[1]}) rotate(${deg})`}>
      <ellipse cx={0} cy={0} rx={rx} ry={rx * 0.16} fill="#d4a93a" {...ink} />
      <ellipse cx={-rx * 0.25} cy={-rx * 0.03} rx={rx * 0.45} ry={rx * 0.05} fill="#f3d77a" opacity={0.8} />
      <circle cx={0} cy={-3} r={5} fill="#8a6a1e" />
    </g>
  </g>
);

export const Drummer: React.FC<{ t: number }> = ({ t }) => {
  const beatPhase = (t % SPB) / SPB;
  const nod = 6 * Math.pow(Math.cos(Math.PI * beatPhase), 2);
  const kick = decay(t, KICK_T, 0.08);
  const hip: V = [960, 700 + kick * 4];
  const torsoDeg = Math.sin(t * 1.7) * 1.5 + kick * 1.5;
  const neck = local(hip, torsoDeg, [0, -170]);
  const headDeg = torsoDeg * 0.5 + nod * 0.5 + ring(t, SONG.drums.filter((d) => d.piece === DRUM.SNARE).map((d) => d.t), 0.15, 2.5, 1.5);
  const head = local(neck, headDeg, [0, -70 + nod * 0.8]);
  const crash = decay(t, CRASH_T, 0.35);

  const shL = local(neck, torsoDeg, [-96, 30]);
  const shR = local(neck, torsoDeg, [96, 30]);
  const L = stick(HANDS.L, t, shL);
  const R = stick(HANDS.R, t, shR);
  const armL = ikArm(shL, L.hand, 100, 96, -1);
  const armR = ikArm(shR, R.hand, 100, 96, 1);

  const hatOpen = SONG.drums.some((d) => d.piece === DRUM.OPEN_HAT && t >= d.t && t < d.t + 0.2);

  return (
    <g>
      <FloorShadow x={960} y={905} rx={330} ry={42} o={0.4} />
      {/* drummer body (behind the kit) */}
      <At p={neck} deg={torsoDeg}>
        <path d="M -98 20 Q -112 120 -92 200 L 92 200 Q 112 120 98 20 Q 0 -10 -98 20 Z" fill={SHIRT} {...ink} />
        <path d="M 98 20 Q 112 120 92 200 L 66 200 Q 84 110 74 26 Z" fill={SHIRT_SH} />
        <path d="M -40 8 Q 0 40 40 8" fill="none" {...ink} />
        <path d="M -30 80 L 30 80 L 0 130 Z" fill="#d9c9a0" opacity={0.85} {...ink} strokeWidth={3} />
      </At>
      <Seg a={local(neck, torsoDeg, [0, 6])} b={local(head, headDeg, [0, 48])} w1={52} w2={48} fill={SKIN} shade={SKIN_SH} />
      <At p={head} deg={headDeg}>
        <ellipse cx={-66} cy={8} rx={14} ry={20} fill={SKIN} {...ink} />
        <ellipse cx={66} cy={8} rx={14} ry={20} fill={SKIN} {...ink} />
        <path d="M -64 -30 Q -70 50 -30 78 Q 0 92 30 78 Q 70 50 64 -30 Z" fill={SKIN} {...ink} />
        <path d="M 64 -30 Q 70 50 30 78 Q 14 86 6 88 Q 50 50 52 -24 Z" fill={SKIN_SH} opacity={0.7} />
        {/* round shades */}
        <circle cx={-26} cy={4} r={20} fill="#151515" {...ink} />
        <circle cx={26} cy={4} r={20} fill="#151515" {...ink} />
        <path d="M -6 2 L 6 2" {...ink} />
        <path d="M -34 -6 Q -28 -10 -22 -6 M 18 -6 Q 24 -10 30 -6" stroke="#9adcff" strokeWidth={3} fill="none" opacity={0.7} />
        <path d="M -2 18 L -9 38 Q 0 42 8 38" fill="none" {...ink} strokeWidth={4} />
        {crash > 0.15 ? (
          <ellipse cx={0} cy={60} rx={11} ry={13 * crash + 4} fill="#3a0f14" {...ink} strokeWidth={4} />
        ) : (
          <path d="M -18 58 Q 0 66 18 56" fill="none" {...ink} strokeWidth={4.5} />
        )}
        {/* stubble */}
        <path d="M -40 56 Q 0 96 40 56" fill="none" stroke="#5a3a28" strokeWidth={10} opacity={0.18} />
        {/* beanie */}
        <path d="M -68 -26 Q -70 -104 0 -108 Q 70 -104 68 -26 Z" fill="#5b3a8a" {...ink} />
        <path d="M -70 -36 L 70 -36 L 70 -14 L -70 -14 Z" fill="#46296e" {...ink} />
        {[-50, -30, -10, 10, 30, 50].map((x) => (
          <line key={x} x1={x} y1={-34} x2={x} y2={-16} stroke={INK} strokeWidth={2} opacity={0.5} />
        ))}
      </At>

      {/* kit */}
      <Cymbal c={[752, 664 - (hatOpen ? 10 : 0)]} rx={70} deg={-4 + ring(t, HAT_T, 0.08, 6, 2)} stand={260} />
      <Cymbal c={[752, 680]} rx={70} deg={-4} />
      <Drum c={[906, 642]} rx={52} ry={16} h={50} hit={decay(t, TOM_T(DRUM.TOM_HI), 0.08)} />
      <Drum c={[1018, 642]} rx={56} ry={17} h={54} hit={decay(t, TOM_T(DRUM.TOM_MID), 0.08)} />
      {/* bass drum with Ben's skull on the front head */}
      <g transform={`translate(960 800) scale(${1 + 0.035 * kick})`}>
        <circle cx={0} cy={0} r={128} fill="#8c1c2a" {...ink} />
        <circle cx={0} cy={0} r={112} fill="#1b1416" {...ink} strokeWidth={3} />
        <image href={staticFile("clip/skull.png")} x={-62} y={-82} width={124} height={166} opacity={0.95} />
        <path d="M -112 0 A 112 112 0 0 0 112 0" fill="none" stroke="#000" strokeWidth={30} opacity={0.15} />
      </g>
      <Drum c={[1122, 752]} rx={68} ry={20} h={110} hit={decay(t, TOM_T(DRUM.TOM_LO), 0.08)} />
      <Drum c={[842, 722]} rx={62} ry={18} h={40} hit={decay(t, SNARE_T, 0.06)} color="#b9bcc4" />
      <line x1={1150} y1={556} x2={1190} y2={900} stroke="#555" strokeWidth={6} />
      <Cymbal c={[1150, 556]} rx={86} deg={8 + ring(t, CRASH_T, 0.45, 2.2, 12)} />

      {/* arms and sticks on top */}
      {[
        { sh: shL, arm: armL, s: L, side: -1 as const },
        { sh: shR, arm: armR, s: R, side: 1 as const },
      ].map(({ sh, arm, s, side }) => (
        <g key={side}>
          <Seg a={sh} b={arm.mid} w1={40} w2={34} fill={SKIN} shade={SKIN_SH} shadeSide={side} />
          <Seg a={arm.mid} b={arm.end} w1={34} w2={28} fill={SKIN} shade={SKIN_SH} shadeSide={side} />
          <line x1={s.hand[0]} y1={s.hand[1]} x2={s.tip[0]} y2={s.tip[1]} stroke={INK} strokeWidth={11} strokeLinecap="round" />
          <line x1={s.hand[0]} y1={s.hand[1]} x2={s.tip[0]} y2={s.tip[1]} stroke="#e2c27c" strokeWidth={6} strokeLinecap="round" />
          <Hand p={s.hand} deg={aim(arm.mid, arm.end)} r={19} fill={SKIN} shade={SKIN_SH} grip />
        </g>
      ))}
    </g>
  );
};
