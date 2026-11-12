import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { GreyCity } from "../backgrounds/GreyCity";
import { Character } from "../lib/Character";
import { Boil, Grain, Vignette } from "../lib/effects";
import { Tear } from "../lib/Tear";

const HAT = { name: "hat-profile", size: [1451, 2023] as [number, number], tearEnd: [807, 1085] as [number, number] };

/** A red balloon drifting up through the sky (drawn in the city's coordinates). */
const Balloon: React.FC = () => {
  const frame = useCurrentFrame();
  const x = 1250 + frame * 1.1 + Math.sin(frame / 18) * 14;
  const y = 720 - frame * 1.9;
  return (
    <g transform={`translate(${x} ${y}) rotate(${Math.sin(frame / 18) * 6})`}>
      <path d="M0 40 Q -6 70 4 100 Q 12 125 0 150" stroke="#141414" strokeWidth={2.5} fill="none" />
      <ellipse cx={0} cy={0} rx={30} ry={38} fill="#b3132e" stroke="#141414" strokeWidth={4} />
      <ellipse cx={-10} cy={-14} rx={7} ry={11} fill="#e8a0a8" opacity={0.7} />
      <path d="M-6 37 L6 37 L0 46 Z" fill="#b3132e" stroke="#141414" strokeWidth={3} />
    </g>
  );
};

export const Test1: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const p = frame / durationInFrames;
  const zoom = interpolate(p, [0, 1], [1, 1.08]);
  const pan = interpolate(p, [0, 1], [0, 160]);

  return (
    <AbsoluteFill style={{ backgroundColor: "#111", filter: "contrast(1.06) saturate(0.92) sepia(0.08)" }}>
      <Boil id="boil" strength={2.2} />
      <Boil id="boilBg" strength={3} every={5} />
      <AbsoluteFill style={{ transform: `scale(${zoom})`, transformOrigin: "22% 38%" }}>
        <AbsoluteFill style={{ filter: "url(#boilBg)", transform: "scale(1.03)" }}>
          <GreyCity pan={pan}>
            <Balloon />
          </GreyCity>
        </AbsoluteFill>
        <Character name={HAT.name} size={HAT.size} x={-30} y={30} height={1090} boilId="boil">
          <Tear at={HAT.tearEnd} run={150} cycle={80} delay={10} />
        </Character>
      </AbsoluteFill>
      <Vignette />
      <Grain />
    </AbsoluteFill>
  );
};
