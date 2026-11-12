import { SONG, SPB, beatTimes, ring, sectionAt } from "./song";
import { At, FloorShadow, Hand, INK, Mouth, MouthShape, Seg, V, add, aim, ik, ikArm, ink, lerp, local } from "./rig";

/**
 * The singer — based on Ben's "Distor": messy brown hair, heavy-lidded eyes with
 * purple shadows, a red tear, big ears, white jacket with grey collar, black shirt.
 * Stands at a mic. Mouth shapes follow the sung notes of the score.
 */

const SKIN = "#f2c39b";
const SKIN_SH = "#d99a72";
const JACKET = "#f3f0e8";
const JACKET_SH = "#c9c3b5";
const JEANS = "#2b3140";
const JEANS_SH = "#1a1e29";
const FOOT_L: V = [585, 1035];
const FOOT_R: V = [700, 1040];
const MIC: V = [742, 548]; // mic head (fixed on its stand)

/** What the voice is doing at time t: mouth shape and openness, plus a smoothed "intensity". */
const voice = (t: number) => {
  const notes = SONG.vocal;
  let shape: MouthShape = "rest";
  let open = 0;
  let pitch = 64;
  let long = 0;
  for (const n of notes) {
    if (t < n.t - 0.04 || t > n.t + n.dur + 0.06) continue;
    const a = Math.min(1, (t - (n.t - 0.04)) / 0.07);
    const r = Math.min(1, (n.t + n.dur + 0.06 - t) / 0.08);
    const vib = n.dur > 0.6 ? 0.08 * Math.sin((t - n.t) * 2 * Math.PI * 5.5) : 0;
    open = Math.max(0, Math.min(a, r) * (0.85 + vib) * ((n as { open?: number }).open ?? 1));
    shape = n.vowel as MouthShape;
    pitch = n.pitch;
    long = n.dur >= 0.9 ? Math.min(1, (t - n.t) / 0.25) * r : 0;
  }
  return { shape, open, pitch, long };
};

/** Smoothly raise the free arm on long chorus notes. */
const reach = (t: number) => {
  let v = 0;
  for (const n of SONG.vocal) {
    if (n.dur < 0.45 || sectionAt(n.t) !== "chorus") continue;
    const up = Math.min(1, Math.max(0, (t - n.t + 0.15) / 0.35));
    const down = Math.min(1, Math.max(0, (n.t + n.dur + 0.5 - t) / 0.4));
    v = Math.max(v, Math.min(up, down));
  }
  return v * v * (3 - 2 * v);
};

const lastBlink = (t: number) => {
  // a blink every ~3.3 s (varied), 0.12 s long
  const period = 3.3;
  const k = Math.floor(t / period);
  const start = k * period + ((k * 7919) % 10) / 10;
  const dt = t - start;
  return dt >= 0 && dt < 0.13 ? 1 - Math.abs(dt - 0.065) / 0.065 : 0;
};

export const Singer: React.FC<{ t: number }> = ({ t }) => {
  const sec = sectionAt(t);
  const v = voice(t);
  const beatPhase = (t % SPB) / SPB;
  const bounce = (sec === "chorus" ? 9 : 5) * Math.pow(Math.cos(Math.PI * beatPhase), 2); // dip on each beat
  const sway = Math.sin((t / (SPB * 4)) * Math.PI * 2) * 3;

  // hips and torso
  const hip: V = [640 + sway * 2, 786 + bounce];
  const torsoDeg = sway * 0.8 + (sec === "chorus" ? -2 : 0) + v.long * -3;
  const neck = local(hip, torsoDeg, [0, -205]);
  const headDeg = torsoDeg * 0.6 + ring(t, beatTimes, 0.18, 2.2, 2.5) - (v.pitch - 66) * 0.5 + v.long * -4;
  const head = local(neck, headDeg, [0, -78]);

  // legs: knees bend as the hips dip
  const hipL = local(hip, torsoDeg, [-34, 6]);
  const hipR = local(hip, torsoDeg, [34, 6]);
  const legL = ik(hipL, add(FOOT_L, [0, -22]), 118, 118, 1);
  const legR = ik(hipR, add(FOOT_R, [0, -22]), 118, 118, -1);

  // arms: right hand (screen-left) holds the mic stand, left hand gestures
  const shL = local(neck, torsoDeg, [-74, 22]);
  const shR = local(neck, torsoDeg, [74, 22]);
  const standGrip: V = [MIC[0] + 14, MIC[1] + 150];
  const armL = ikArm(shL, standGrip, 105, 98, -1);
  const r = reach(t);
  const restHand: V = local(hip, torsoDeg, [92, 20]);
  const upHand: V = [shR[0] + 120, shR[1] - 40 + ring(t, beatTimes, 0.2, 2, 6)];
  const armR = ikArm(shR, lerp(restHand, upHand, r), 105, 98, 1);

  const blink = Math.max(lastBlink(t), v.long * 0.85);
  const hair = ring(t, beatTimes, 0.22, 2.4, 3);

  return (
    <g>
      <FloorShadow x={645} y={1042} rx={150} />
      {/* mic stand (behind the arm) */}
      <path d={`M ${MIC[0] + 40} 1045 L ${MIC[0] + 8} ${MIC[1] + 30}`} {...ink} stroke="#2a2a2a" strokeWidth={9} />
      <path d={`M ${MIC[0] + 40} 1045 L ${MIC[0] + 8} ${MIC[1] + 30}`} fill="none" stroke="#6a6a6a" strokeWidth={3} />
      <path d={`M ${MIC[0] - 20} 1048 L ${MIC[0] + 100} 1048 M ${MIC[0] + 40} 1045 L ${MIC[0] + 85} 1030`} {...ink} strokeWidth={7} />

      {/* legs */}
      <Seg a={hipL} b={legL.mid} w1={58} w2={46} fill={JEANS} shade={JEANS_SH} />
      <Seg a={legL.mid} b={legL.end} w1={46} w2={38} fill={JEANS} shade={JEANS_SH} />
      <Seg a={hipR} b={legR.mid} w1={58} w2={46} fill={JEANS} shade={JEANS_SH} />
      <Seg a={legR.mid} b={legR.end} w1={46} w2={38} fill={JEANS} shade={JEANS_SH} />
      <path d={`M ${FOOT_L[0] - 42} ${FOOT_L[1]} Q ${FOOT_L[0] - 40} ${FOOT_L[1] - 30} ${FOOT_L[0]} ${FOOT_L[1] - 28} L ${FOOT_L[0] + 18} ${FOOT_L[1] - 26} L ${FOOT_L[0] + 20} ${FOOT_L[1]} Z`} fill="#141414" {...ink} />
      <path d={`M ${FOOT_R[0] + 46} ${FOOT_R[1]} Q ${FOOT_R[0] + 44} ${FOOT_R[1] - 30} ${FOOT_R[0]} ${FOOT_R[1] - 28} L ${FOOT_R[0] - 18} ${FOOT_R[1] - 26} L ${FOOT_R[0] - 20} ${FOOT_R[1]} Z`} fill="#141414" {...ink} />

      {/* torso: black shirt with red flames under an open white jacket */}
      <At p={neck} deg={torsoDeg}>
        <path d="M -78 18 L -66 200 L 66 200 L 78 18 Q 0 -6 -78 18 Z" fill="#1c1b1e" {...ink} />
        <path d="M -42 120 L -30 92 L -18 116 L -4 80 L 10 112 L 24 86 L 38 118 L 44 200 L -46 200 Z" fill="#c4122f" />
        <path d="M -40 150 L -28 128 L -14 148 L 2 124 L 16 146 L 32 126 L 42 150 L 44 200 L -44 200 Z" fill="#5a5f68" />
        {/* jacket panels */}
        <path d="M -80 16 Q -96 110 -78 214 L -34 214 L -40 60 L -22 10 Z" fill={JACKET} {...ink} />
        <path d="M 80 16 Q 96 110 78 214 L 34 214 L 40 60 L 22 10 Z" fill={JACKET} {...ink} />
        <path d="M 80 16 Q 96 110 78 214 L 62 214 Q 74 110 66 30 Z" fill={JACKET_SH} />
        <path d="M -34 214 L -40 60 L -22 10" fill="none" {...ink} />
        {/* stand-up collar */}
        <path d="M -42 8 L -30 -22 L -10 -6 L -22 18 Z" fill={JACKET} {...ink} />
        <path d="M 42 8 L 30 -22 L 10 -6 L 22 18 Z" fill={JACKET} {...ink} />
        <path d="M -30 -22 L -10 -6 M 30 -22 L 10 -6" stroke="#a5a8b5" strokeWidth={6} />
        <circle cx={-34} cy={96} r={7} fill="#e3b21f" {...ink} strokeWidth={3} />
        <circle cx={-36} cy={170} r={7} fill="#e3b21f" {...ink} strokeWidth={3} />
      </At>

      {/* neck */}
      <Seg a={local(neck, torsoDeg, [0, 6])} b={local(head, headDeg, [0, 52])} w1={40} w2={36} fill={SKIN} shade={SKIN_SH} />

      {/* head */}
      <At p={head} deg={headDeg}>
        {/* ears */}
        <ellipse cx={-64} cy={2} rx={17} ry={27} fill={SKIN} {...ink} />
        <ellipse cx={64} cy={2} rx={17} ry={27} fill={SKIN} {...ink} />
        <path d="M -66 -10 Q -74 2 -64 16 M 66 -10 Q 74 2 64 16" fill="none" {...ink} strokeWidth={3} />
        {/* face: long, strong jaw */}
        <path d="M -58 -44 Q -62 26 -44 64 L -6 92 L 6 92 L 44 64 Q 62 26 58 -44 Q 46 -98 0 -100 Q -46 -98 -58 -44 Z" fill={SKIN} {...ink} />
        <path d="M 58 -44 Q 62 26 44 64 L 6 92 L 30 70 Q 50 30 46 -30 Z" fill={SKIN_SH} opacity={0.7} />
        {/* eyes: big, heavy-lidded, purple underneath, no pupils */}
        {[-1, 1].map((s) => (
          <g key={s} transform={`translate(${s * 27} -10)`}>
            <path d="M -24 0 Q 0 -10 24 0 Q 18 14 0 15 Q -18 14 -24 0 Z" fill="#eceaf0" />
            <path d="M -21 10 Q 0 20 21 10" fill="none" stroke="#5b2c7d" strokeWidth={5} strokeLinecap="round" />
            <path d={`M -24 0 Q 0 -10 24 0 L 24 ${2 + blink * 14} Q 0 ${-6 + blink * 22} -24 ${2 + blink * 14} Z`} fill={SKIN} />
            <path d={`M -26 ${1 + blink * 13} Q 0 ${-11 + blink * 22} 26 ${1 + blink * 13}`} fill="none" {...ink} strokeWidth={6.5} />
            <path d="M -18 13 Q 0 17 18 13" fill="none" {...ink} strokeWidth={2.5} />
          </g>
        ))}
        {/* nose: broad, flat underside */}
        <path d="M -4 -4 L -15 34 Q 0 42 15 34 L 8 20" fill="none" {...ink} strokeWidth={4} />
        <path d="M -15 34 Q 0 42 15 34 L 8 20 L 2 30 Z" fill={SKIN_SH} opacity={0.6} />
        {/* tear */}
        <path d="M -38 6 C -36 30 -40 52 -36 70" fill="none" stroke="#c4122f" strokeWidth={5} strokeLinecap="round" />
        <path d="M -24 8 C -22 22 -24 34 -22 44" fill="none" stroke="#c4122f" strokeWidth={4} strokeLinecap="round" />
        {/* mouth */}
        <g transform="translate(0 60)">
          <Mouth shape={v.shape} open={v.open} w={36} />
        </g>
        {/* hair: Ben's hair texture, messy curled clumps, fringe over the eyes */}
        <g transform={`translate(0 ${-hair * 0.3}) rotate(${hair * 0.6})`}>
          <path
            d="M -70 -6 Q -86 -40 -66 -66 Q -84 -82 -60 -100 Q -66 -128 -34 -122 Q -26 -150 4 -132 Q 22 -156 42 -128 Q 76 -136 70 -104 Q 96 -92 76 -64 Q 92 -40 72 -10 Q 66 -34 58 -40 L 52 -18 Q 44 -34 38 -40 L 30 -14 Q 22 -34 16 -40 L 6 -12 Q -2 -32 -8 -38 L -18 -12 Q -24 -32 -32 -38 L -40 -14 Q -46 -32 -52 -36 L -60 -8 Z"
            fill="#7a4a2c"
            {...ink}
          />
          <path d="M -52 -70 Q -40 -60 -36 -40 M -20 -100 Q -10 -76 -12 -50 M 20 -110 Q 30 -84 26 -54 M 52 -86 Q 56 -66 48 -46" fill="none" {...ink} strokeWidth={3.5} />
          <path d="M -60 -50 Q -54 -64 -44 -70 M -40 -96 Q -26 -100 -18 -94 M 4 -116 Q 16 -118 24 -110 M 46 -100 Q 58 -96 62 -84" fill="none" stroke="#e9d6b4" strokeWidth={5} strokeLinecap="round" opacity={0.85} />
        </g>
      </At>

      {/* mic */}
      <g transform={`translate(${MIC[0]} ${MIC[1]}) rotate(-28)`}>
        <rect x={-8} y={10} width={16} height={60} rx={6} fill="#222" {...ink} strokeWidth={4} />
        <ellipse cx={0} cy={0} rx={17} ry={20} fill="#9a9a9a" {...ink} strokeWidth={4} />
        <path d="M -12 -6 L 12 -6 M -14 2 L 14 2 M -12 10 L 12 10" stroke={INK} strokeWidth={2} />
      </g>

      {/* arms (in front) */}
      <Seg a={shL} b={armL.mid} w1={50} w2={42} fill={JACKET} shade={JACKET_SH} shadeSide={-1} />
      <Seg a={armL.mid} b={armL.end} w1={42} w2={34} fill={JACKET} shade={JACKET_SH} shadeSide={-1} />
      <Hand p={armL.end} deg={aim(armL.mid, armL.end)} fill={SKIN} shade={SKIN_SH} grip />
      <Seg a={shR} b={armR.mid} w1={50} w2={42} fill={JACKET} shade={JACKET_SH} />
      <Seg a={armR.mid} b={armR.end} w1={42} w2={34} fill={JACKET} shade={JACKET_SH} />
      <Hand p={armR.end} deg={aim(armR.mid, armR.end) + r * 40} fill={SKIN} shade={SKIN_SH} grip={r < 0.3} />
    </g>
  );
};

export const voiceAt = voice;
