import { AbsoluteFill, Audio, continueRender, delayRender, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Boil, Grain, Vignette } from "../lib/effects";
import { Drummer } from "./Drummer";
import { Guitarist } from "./Guitarist";
import { Singer } from "./Singer";
import { DRUM, FPS, SONG, SPB, barTimes, beatTimes, decay, onTwos, sectionAt } from "./song";
import { Amp, Beams, StageBack } from "./Stage";

const fontHandle = delayRender("band font");
new FontFace("MetalMania", `url(${staticFile("fonts/metal-mania-latin-400-normal.woff2")})`)
  .load()
  .then((f) => {
    (document.fonts as unknown as { add: (f: FontFace) => void }).add(f);
    continueRender(fontHandle);
  })
  .catch(() => continueRender(fontHandle));

export const BAND_FRAMES = Math.round(SONG.duration * FPS);

/** Cameras: [start sec, x, y, zoom, drift-x, drift-zoom]. x/y = world point at frame center. Cuts on bars. */
type Cam = { at: number; x: number; y: number; z: number; dx?: number; dz?: number };
const B = SPB * 4;
const SHOTS: Cam[] = [
  { at: 0, x: 960, y: 560, z: 1.0, dz: 0.06 }, // intro: wide, slow push
  { at: 2 * B, x: 660, y: 520, z: 2.0, dx: 20 }, // verse: singer
  { at: 4 * B, x: 960, y: 600, z: 1.15, dx: -60 }, // wide
  { at: 6 * B, x: 960, y: 560, z: 1.9 }, // drummer
  { at: 8 * B, x: 650, y: 470, z: 2.6, dz: 0.15 }, // singer close
  { at: 10 * B, x: 1320, y: 640, z: 1.8, dx: 30 }, // guitarist; verse ends with a fill
  { at: 11 * B, x: 960, y: 600, z: 2.1 }, // drummer fill
  { at: 12 * B, x: 960, y: 560, z: 1.0, dz: 0.05 }, // chorus: wide
  { at: 14 * B, x: 650, y: 485, z: 3.0, dz: 0.08 }, // singer, tight on the face
  { at: 16 * B, x: 1300, y: 600, z: 1.7, dx: -40 }, // guitarist
  { at: 18 * B, x: 760, y: 560, z: 1.35, dx: 120 }, // pan across the band
  { at: 20 * B, x: 960, y: 560, z: 1.9 }, // outro: drummer fill
  { at: 21 * B, x: 960, y: 560, z: 1.0, dz: -0.03 }, // last hit, wide
];

const camera = (t: number) => {
  let i = 0;
  for (let k = 0; k < SHOTS.length; k++) if (SHOTS[k].at <= t) i = k;
  const c = SHOTS[i];
  const end = SHOTS[i + 1]?.at ?? SONG.duration;
  const u = (t - c.at) / (end - c.at);
  const ease = u * u * (3 - 2 * u);
  // a little handheld drift
  const hx = Math.sin(t * 1.3) * 4 + Math.sin(t * 2.9) * 2;
  const hy = Math.cos(t * 1.1) * 3 + Math.sin(t * 3.7) * 1.5;
  return { x: c.x + (c.dx ?? 0) * ease + hx, y: c.y + hy, z: c.z * (1 + (c.dz ?? 0) * ease) };
};

const crashes = SONG.drums.filter((d) => d.piece === DRUM.CRASH).map((d) => d.t);
const kicks = SONG.drums.filter((d) => d.piece === DRUM.KICK).map((d) => d.t);
const WASH = ["#ff3d6e", "#3dd6ff", "#c86bff", "#ffb13d"];

export const BandVideo: React.FC = () => {
  const frame = useCurrentFrame();
  const t = frame / FPS; // camera and lights: smooth
  const tp = onTwos(frame); // characters: on twos
  const sec = sectionAt(t);
  const cam = camera(t);
  const bar = Math.floor(t / B);
  const wash = WASH[bar % WASH.length];
  const washAmt = sec === "chorus" ? 0.35 + 0.5 * decay(t, beatTimes, 0.18) : sec === "outro" ? 0.5 * decay(t, crashes, 0.5) : 0.12 + 0.15 * decay(t, barTimes, 0.3);
  const flash = decay(t, crashes, 0.1);
  const fadeOut = interpolate(t, [SONG.duration - 2.2, SONG.duration], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{ backgroundColor: "#0c0b0a" }}>
      <Audio src={staticFile("song/song.wav")} />
      <Boil id="boil" strength={2} every={2} />
      <AbsoluteFill style={{ opacity: fadeOut, filter: "url(#boil) contrast(1.05) saturate(1.05)" }}>
        <svg viewBox="0 0 1920 1080" width="100%" height="100%">
          <defs>
            <pattern id="hairTex" patternUnits="userSpaceOnUse" width={420} height={124} patternTransform="rotate(-6)">
              <image href={staticFile("tex/hair.png")} width={420} height={124} preserveAspectRatio="none" />
            </pattern>
            <filter id="softShadow" x="-50%" y="-200%" width="200%" height="500%">
              <feGaussianBlur stdDeviation={10} />
            </filter>
            <radialGradient id="spot">
              <stop offset="0" stopColor="#000" />
              <stop offset="0.55" stopColor="#000" stopOpacity={0.85} />
              <stop offset="1" stopColor="#000" stopOpacity={0} />
            </radialGradient>
            <mask id="spots" maskUnits="userSpaceOnUse" x={-400} y={-400} width={2720} height={1880}>
              <rect x={-400} y={-400} width={2720} height={1880} fill="#fff" />
              <ellipse cx={650} cy={700} rx={300} ry={420} fill="url(#spot)" />
              <ellipse cx={960} cy={640} rx={330} ry={360} fill="url(#spot)" />
              <ellipse cx={1330} cy={720} rx={300} ry={420} fill="url(#spot)" />
              <ellipse cx={560} cy={200} rx={260} ry={200} fill="url(#spot)" />
            </mask>
            <pattern id="paper" patternUnits="userSpaceOnUse" width={300} height={380}>
              <image href={staticFile("tex/jacket.png")} width={300} height={380} preserveAspectRatio="none" />
            </pattern>
            <filter id="beamBlur">
              <feGaussianBlur stdDeviation={18} />
            </filter>
          </defs>
          <g transform={`translate(960 540) scale(${cam.z}) translate(${-cam.x} ${-cam.y})`}>
            <StageBack glow={decay(t, kicks, 0.15)} wash={wash} washAmt={washAmt} />
            <Amp x={150} y={430} w={250} h={300} thump={decay(t, kicks, 0.1)} />
            <Amp x={1540} y={400} w={270} h={330} thump={decay(t, kicks, 0.1)} />
            <Drummer t={tp} />
            <Guitarist t={tp} />
            <Singer t={tp} />
            <Beams color={wash} amt={sec === "chorus" ? 1 : 0.35} />
            <rect x={-400} y={-400} width={2720} height={1880} fill="#0b0912" opacity={0.5 - 0.12 * decay(t, kicks, 0.12)} mask="url(#spots)" />
          </g>
          <rect width={1920} height={1080} fill="url(#paper)" opacity={0.22} style={{ mixBlendMode: "multiply" }} />
          <rect width={1920} height={1080} fill="#fff" opacity={flash * 0.18} />
        </svg>
      </AbsoluteFill>
      <Vignette strength={0.5} />
      <Grain opacity={0.12} />
    </AbsoluteFill>
  );
};
