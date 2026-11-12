import { staticFile } from "remotion";
import { INK, ink } from "./rig";

/** The rehearsal room: back wall, posters, bulb, wooden floor, rug, amps. World 1920 x 1080. */

const WALL = 640; // y where the wall meets the floor
const VP = [960, 330]; // vanishing point for the floor planks

export const StageBack: React.FC<{ glow: number; wash: string; washAmt: number }> = ({ glow, wash, washAmt }) => {
  const bricks: React.ReactNode[] = [];
  for (let r = 0; r < 16; r++) {
    const y = r * 40;
    bricks.push(<line key={`h${r}`} x1={0} y1={y} x2={1920} y2={y} stroke={INK} strokeWidth={2} opacity={0.18} />);
    for (let c = 0; c < 22; c++) {
      const x = c * 90 + (r % 2 ? 45 : 0);
      bricks.push(<line key={`v${r}-${c}`} x1={x} y1={y} x2={x} y2={y + 40} stroke={INK} strokeWidth={2} opacity={0.14} />);
    }
  }
  const planks: React.ReactNode[] = [];
  for (let i = -14; i <= 14; i++) {
    const xb = 960 + i * 70;
    const xf = VP[0] + (xb - VP[0]) * ((1080 - VP[1]) / (WALL - VP[1]));
    planks.push(<line key={`p${i}`} x1={xb} y1={WALL} x2={xf} y2={1080} stroke={INK} strokeWidth={2.5} opacity={0.35} />);
  }
  return (
    <g>
      <defs>
        <radialGradient id="bulbGlow">
          <stop offset="0" stopColor="#ffe9a8" stopOpacity={0.55} />
          <stop offset="1" stopColor="#ffe9a8" stopOpacity={0} />
        </radialGradient>
        <radialGradient id="wash" cx="0.5" cy="0.35" r="0.6">
          <stop offset="0" stopColor={wash} stopOpacity={1} />
          <stop offset="1" stopColor={wash} stopOpacity={0} />
        </radialGradient>
        <linearGradient id="floorFade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#000" stopOpacity={0.45} />
          <stop offset="0.5" stopColor="#000" stopOpacity={0.1} />
          <stop offset="1" stopColor="#000" stopOpacity={0.3} />
        </linearGradient>
      </defs>
      {/* wall */}
      <rect width={1920} height={WALL} fill="#36565a" />
      {bricks}
      <rect width={1920} height={WALL} fill="url(#wash)" opacity={washAmt} style={{ mixBlendMode: "screen" }} />
      {/* posters */}
      <g transform="translate(330 120) rotate(-4)">
        <rect width={230} height={320} fill="#c9b98f" {...ink} />
        <rect x={14} y={14} width={202} height={292} fill="#2b1d33" />
        <image href={staticFile("clip/skull.png")} x={40} y={110} width={150} height={200} />
        <text x={115} y={66} textAnchor="middle" fontFamily="MetalMania" fontSize={38} fill="#d9c9a0">DISTORTED</text>
        <text x={115} y={104} textAnchor="middle" fontFamily="MetalMania" fontSize={38} fill="#d9c9a0">REALITY</text>
      </g>
      <g transform="translate(1260 90) rotate(3)">
        <rect width={200} height={270} fill="#e3d6b6" {...ink} />
        <circle cx={100} cy={110} r={58} fill="#b3132e" {...ink} />
        <path d="M100 168 Q 92 200 104 240" fill="none" {...ink} strokeWidth={3} />
        <text x={100} y={258} textAnchor="middle" fontFamily="MetalMania" fontSize={26} fill={INK}>LIVE TONIGHT</text>
      </g>
      <g transform="translate(1660 200) rotate(-2)">
        <rect width={150} height={110} fill="#8fa38a" {...ink} />
        <path d="M 10 95 L 45 50 L 70 75 L 105 30 L 140 95 Z" fill="#4c5e48" />
      </g>
      {/* floor */}
      <rect y={WALL} width={1920} height={1080 - WALL} fill="#7a5233" />
      {planks}
      <line x1={0} y1={WALL} x2={1920} y2={WALL} stroke={INK} strokeWidth={5} />
      <rect y={WALL} width={1920} height={1080 - WALL} fill="url(#floorFade)" />
      {/* rug under the front */}
      <path d="M 330 860 L 1650 860 L 1840 1080 L 120 1080 Z" fill="#6b2430" {...ink} />
      <path d="M 380 880 L 1600 880 L 1770 1060 L 190 1060 Z" fill="none" stroke="#d9a441" strokeWidth={6} opacity={0.6} />
      {/* bulb */}
      <line x1={560} y1={0} x2={560} y2={170} stroke={INK} strokeWidth={4} />
      <circle cx={560} cy={180} r={300} fill="url(#bulbGlow)" opacity={0.6 + 0.4 * glow} />
      <circle cx={560} cy={185} r={20} fill="#fff4c8" {...ink} />
    </g>
  );
};

/** Amp cabinet (front view). */
export const Amp: React.FC<{ x: number; y: number; w: number; h: number; thump?: number }> = ({ x, y, w, h, thump = 0 }) => (
  <g transform={`translate(${x} ${y})`}>
    <rect width={w} height={h} rx={8} fill="#1f1d1c" {...ink} />
    <rect x={w * 0.06} y={h * 0.08} width={w * 0.88} height={h * 0.16} rx={4} fill="#2f2c2a" />
    {[0.2, 0.35, 0.5, 0.65, 0.8].map((k) => (
      <circle key={k} cx={w * k} cy={h * 0.16} r={h * 0.035} fill="#d9c9a0" />
    ))}
    <rect x={w * 0.06} y={h * 0.3} width={w * 0.88} height={h * 0.64} fill="#3b3631" stroke={INK} strokeWidth={3} />
    {[0.3, 0.7].map((k) => (
      <circle key={k} cx={w * k} cy={h * 0.62} r={w * 0.17 * (1 + 0.04 * thump)} fill="#171514" stroke="#5a524a" strokeWidth={4} />
    ))}
    <text x={w * 0.5} y={h * 0.27} textAnchor="middle" fontFamily="MetalMania" fontSize={h * 0.08} fill="#d9a441">
      MARSHMALLOW
    </text>
  </g>
);

/** Colored light beams from the ceiling (stage lights). */
export const Beams: React.FC<{ color: string; amt: number }> = ({ color, amt }) => (
  <g opacity={amt} style={{ mixBlendMode: "screen" }}>
    {[
      [420, 0, 260, 1080, 760, 1080],
      [1500, 0, 1180, 1080, 1700, 1080],
      [960, 0, 760, 1080, 1160, 1080],
    ].map((p, i) => (
      <polygon key={i} points={`${p[0] - 30},0 ${p[0] + 30},0 ${p[4]},${p[5]} ${p[2]},${p[3]}`} fill={color} opacity={0.18} filter="url(#beamBlur)" />
    ))}
  </g>
);
