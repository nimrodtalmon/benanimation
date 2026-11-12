import { SONG, SPB, around, beatTimes, ring, sectionAt } from "./song";
import { At, FloorShadow, Hand, INK, Seg, V, add, aim, ik, ink, local } from "./rig";

/**
 * The guitarist — based on Ben's hat character: orange hair, white spiked hat with
 * a cross, green eyes with dark lids, red tears, dark high-collar coat.
 * Every strum and chord change comes from the score.
 */

const SKIN = "#f0c3a0";
const SKIN_SH = "#d4977a";
const COAT = "#2c2e36";
const COAT_SH = "#1b1c22";
const PANTS = "#3b3a40";
const PANTS_SH = "#26252a";
const FOOT_L: V = [1250, 1032];
const FOOT_R: V = [1415, 1032];

/** Where the strumming hand is across the strings (guitar-local y), from the strum events. */
const strumY = (t: number) => {
  const { prev, next } = around(SONG.guitar, t);
  const amp = (s?: { accent: boolean }) => (sectionAt(t) === "chorus" ? 34 : 16) * (s?.accent ? 1.25 : 1);
  const startOf = (s: { dir: string; accent: boolean }) => (s.dir === "down" ? -amp(s) : amp(s));
  if (!prev) return next ? startOf(next) : -20;
  const end = -startOf(prev);
  const x = t - prev.t;
  if (x < 0.06) {
    const u = x / 0.06;
    return startOf(prev) + (end - startOf(prev)) * u * u * (3 - 2 * u);
  }
  if (!next || next.t - prev.t > 1.2) return end + Math.sin(x * 3) * 3; // let the chord ring
  const gap = next.t - prev.t;
  const u = Math.min(1, Math.max(0, (x - 0.06) / (gap - 0.06)));
  return end + (startOf(next) - end) * u * u * (3 - 2 * u);
};

/** Fretting-hand position along the neck (guitar-local x) for the current chord. */
const FRET: Record<string, number> = { Em: 318, G: 292, C: 282, D: 258 };
const fretX = (t: number) => {
  const { prev, next } = around(SONG.guitar, t);
  const cur = FRET[prev?.chord ?? "Em"];
  if (next && next.chord !== prev?.chord && next.t - t < 0.12) {
    const u = 1 - (next.t - t) / 0.12;
    return cur + (FRET[next.chord] - cur) * u;
  }
  return cur;
};

export const Guitarist: React.FC<{ t: number }> = ({ t }) => {
  const sec = sectionAt(t);
  const beatPhase = (t % SPB) / SPB;
  const beatNo = Math.floor(t / SPB) % 4;
  const hard = sec === "chorus";
  const bounce = (hard ? 12 : 6) * Math.pow(Math.sin(Math.PI * beatPhase), 2);
  const hip: V = [1332, 784 + bounce];
  const torsoDeg = -3 + Math.sin((t / (SPB * 8)) * Math.PI * 2) * 2 + (hard ? 4 * Math.pow(Math.sin(Math.PI * beatPhase), 2) : 0);
  const neck = local(hip, torsoDeg, [0, -200]);
  const bang = hard ? 13 * Math.pow(Math.sin(Math.PI * beatPhase), 3) : beatNo % 2 === 0 ? 5 * Math.pow(Math.sin(Math.PI * beatPhase), 3) : 0;
  const headDeg = torsoDeg + bang * 0.4 + ring(t, beatTimes, 0.2, 2.4, 2);
  const headDrop = bang * 1.2;
  const head = local(neck, headDeg, [0, -74 + headDrop]);

  const hipL = local(hip, torsoDeg, [-40, 4]);
  const hipR = local(hip, torsoDeg, [40, 4]);
  const legL = ik(hipL, add(FOOT_L, [0, -22]), 120, 120, 1);
  const legR = ik(hipR, add(FOOT_R, [0, -22]), 120, 120, -1);

  // guitar hangs in front of the hips, neck up to screen-right
  const gAt = local(hip, torsoDeg, [-6, -22]);
  const gDeg = -24 + torsoDeg * 0.8 + ring(t, beatTimes, 0.25, 1.6, hard ? 2 : 1);
  const G = (p: V) => local(gAt, gDeg, p);

  const shL = local(neck, torsoDeg, [-72, 24]); // strumming arm (screen-left)
  const shR = local(neck, torsoDeg, [72, 24]); // fretting arm (screen-right)
  const strum = G([-6, strumY(t)]);
  const armL = ik(shL, add(strum, [-10, -14]), 104, 100, 1);
  const fret = G([fretX(t), 26]);
  const armR = ik(shR, fret, 104, 104, 1);
  const hair = ring(t, beatTimes, 0.2, 2.6, hard ? 6 : 3);

  return (
    <g>
      <FloorShadow x={1335} y={1040} rx={170} />
      {/* legs */}
      <Seg a={hipL} b={legL.mid} w1={56} w2={46} fill={PANTS} shade={PANTS_SH} />
      <Seg a={legL.mid} b={legL.end} w1={46} w2={38} fill={PANTS} shade={PANTS_SH} />
      <Seg a={hipR} b={legR.mid} w1={56} w2={46} fill={PANTS} shade={PANTS_SH} />
      <Seg a={legR.mid} b={legR.end} w1={46} w2={38} fill={PANTS} shade={PANTS_SH} />
      {[FOOT_L, FOOT_R].map((f, i) => (
        <path key={i} d={`M ${f[0] - 34} ${f[1] + 2} L ${f[0] - 30} ${f[1] - 30} L ${f[0] + 26} ${f[1] - 30} Q ${f[0] + 44} ${f[1] - 16} ${f[0] + 40} ${f[1] + 2} Z`} fill="#3a2418" {...ink} />
      ))}

      {/* back arm (fretting) is behind the guitar neck */}
      <Seg a={shR} b={armR.mid} w1={48} w2={40} fill={COAT} shade={COAT_SH} />
      <Seg a={armR.mid} b={armR.end} w1={40} w2={32} fill={COAT} shade={COAT_SH} />

      {/* torso: long dark coat with a high collar */}
      <At p={neck} deg={torsoDeg}>
        <path d="M -80 18 Q -94 120 -86 230 L 86 230 Q 94 120 80 18 Q 0 -8 -80 18 Z" fill={COAT} {...ink} />
        <path d="M 80 18 Q 94 120 86 230 L 60 230 Q 72 120 62 26 Z" fill={COAT_SH} />
        <path d="M -24 16 L 0 110 L 24 16" fill="#8a1c24" {...ink} />
        <path d="M -48 -26 L -24 16 L 0 4 L 24 16 L 48 -26 L 54 14 L 0 34 L -54 14 Z" fill={COAT} {...ink} />
        <path d="M 0 110 L 0 230" {...ink} />
      </At>

      {/* neck + head */}
      <Seg a={local(neck, torsoDeg, [0, 4])} b={local(head, headDeg, [0, 48])} w1={36} w2={32} fill={SKIN} shade={SKIN_SH} />
      <At p={head} deg={headDeg}>
        {/* orange hair sticking out under the hat */}
        <g transform={`rotate(${hair * 0.5})`}>
          <path d="M -66 -40 Q -92 -10 -76 30 Q -90 52 -64 66 L -50 20 Z M 66 -40 Q 92 -10 76 30 Q 90 52 64 66 L 50 20 Z" fill="#e0662a" {...ink} />
        </g>
        {/* face: narrower, pointed chin */}
        <path d="M -54 -40 Q -60 30 -32 70 Q -12 92 0 94 Q 12 92 32 70 Q 60 30 54 -40 Z" fill={SKIN} {...ink} />
        <path d="M 54 -40 Q 60 30 32 70 Q 14 90 4 92 Q 36 60 42 -30 Z" fill={SKIN_SH} opacity={0.7} />
        {/* eyes: green, dark heavy lids */}
        {[-1, 1].map((s) => (
          <g key={s} transform={`translate(${s * 24} -2)`}>
            <path d="M -18 0 Q 0 -9 18 0 Q 12 11 0 12 Q -12 11 -18 0 Z" fill="#f4f1ea" {...ink} strokeWidth={3} />
            <circle cx={s * 3} cy={3} r={6.5} fill="#2f9a4a" stroke={INK} strokeWidth={2.5} />
            <circle cx={s * 3} cy={3} r={2.5} fill={INK} />
            <path d="M -20 1 Q 0 -12 20 1" fill="none" {...ink} strokeWidth={6} />
            <path d="M -16 14 Q 0 19 16 14" fill="none" stroke="#7a5a8a" strokeWidth={3} />
          </g>
        ))}
        {/* red tears on one side */}
        <path d="M -30 12 C -28 32 -32 50 -28 66 M -18 12 C -16 24 -18 34 -16 42" fill="none" stroke="#c4122f" strokeWidth={4.5} strokeLinecap="round" />
        <path d="M -2 8 L -10 38 Q 0 44 8 38" fill="none" {...ink} strokeWidth={4} />
        <path d={`M -18 64 Q 0 ${hard ? 72 : 66} 18 62`} fill={hard ? "#3a0f14" : "none"} {...ink} strokeWidth={4} />
        {/* white hat with grey spikes and a cross */}
        <path d="M -70 -34 Q -74 -96 0 -104 Q 74 -96 70 -34 Q 0 -50 -70 -34 Z" fill="#f4f2ec" {...ink} />
        <path d="M -70 -34 Q 0 -50 70 -34 L 72 -20 Q 0 -36 -72 -20 Z" fill="#a9acb6" {...ink} />
        <path d="M -40 -94 L -52 -132 L -20 -100 Z M 8 -104 L 6 -146 L 28 -102 Z M 46 -94 L 70 -124 L 58 -84 Z" fill="#8e929c" {...ink} />
        <path d="M 0 -88 L 0 -56 M -14 -74 L 14 -74" {...ink} strokeWidth={5} />
      </At>

      {/* guitar */}
      <At p={gAt} deg={gDeg}>
        <path d="M 0 0" />
        <line x1={-60} y1={-40} x2={0} y2={0} stroke="none" />
        {/* body */}
        <path d="M -95 -10 Q -110 -62 -60 -66 Q -30 -64 -10 -46 Q 20 -66 50 -52 Q 70 -40 60 -16 Q 80 10 60 36 Q 70 56 46 62 Q 10 70 -10 50 Q -40 74 -76 58 Q -112 40 -95 -10 Z" fill="#b51f2a" {...ink} />
        <path d="M -76 58 Q -112 40 -95 -10 Q -104 -40 -84 -56 Q -96 0 -64 54 Z" fill="#7d1219" />
        <path d="M -60 -20 Q -30 -36 10 -24 Q 30 0 10 30 Q -30 44 -60 24 Z" fill="#f1e9d8" {...ink} strokeWidth={3} />
        <rect x={-30} y={-22} width={12} height={44} rx={3} fill="#222" />
        <rect x={0} y={-20} width={12} height={40} rx={3} fill="#222" />
        {/* neck and headstock */}
        <rect x={56} y={-11} width={300} height={22} fill="#6b3f22" {...ink} />
        {[100, 140, 176, 208, 236, 262, 286, 308, 330].map((x) => (
          <line key={x} x1={x} y1={-11} x2={x} y2={11} stroke="#d9c9a0" strokeWidth={2} />
        ))}
        <path d="M 356 -14 L 410 -20 L 414 18 L 356 14 Z" fill="#2a2a2a" {...ink} />
        {[-6, -2, 2, 6].map((y) => (
          <line key={y} x1={-40} y1={y} x2={360} y2={y} stroke="#e8e4da" strokeWidth={1.4} opacity={0.9} />
        ))}
        <rect x={-48} y={-14} width={10} height={28} fill="#888" {...ink} strokeWidth={2} />
      </At>
      {/* strap */}
      <path d={`M ${G([60, -50])[0]} ${G([60, -50])[1]} Q ${shR[0]} ${shR[1] - 20} ${shL[0] + 10} ${shL[1] - 4}`} fill="none" stroke="#1a1a1a" strokeWidth={10} />

      {/* fretting hand over the neck */}
      <Hand p={armR.end} deg={aim(armR.mid, armR.end)} fill={SKIN} shade={SKIN_SH} grip />
      {/* strumming arm in front */}
      <Seg a={shL} b={armL.mid} w1={48} w2={40} fill={COAT} shade={COAT_SH} shadeSide={-1} />
      <Seg a={armL.mid} b={armL.end} w1={40} w2={32} fill={COAT} shade={COAT_SH} shadeSide={-1} />
      <Hand p={armL.end} deg={aim(armL.mid, armL.end)} fill={SKIN} shade={SKIN_SH} grip />
    </g>
  );
};
