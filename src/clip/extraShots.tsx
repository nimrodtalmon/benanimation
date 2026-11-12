import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { beatPulse } from "./music";

/** Shots made from Ben's own video: the title and the painted skull. */

export const TINTS = {
  magenta: ["#2a0638", "#d21a9e", "#f2b6e2"],
  green: ["#0f2a05", "#3fd11d", "#d8f7c2"],
  blue: ["#0a0f3a", "#3b3bea", "#c9c6ff"],
  red: ["#2a0505", "#c4122f", "#f3b0b0"],
} as const;
export type Tint = keyof typeof TINTS;

const Smear: React.FC<{ tint: Tint }> = ({ tint }) => {
  const [dark, mid, light] = TINTS[tint];
  return <AbsoluteFill style={{ background: `linear-gradient(180deg, ${dark} 0%, ${dark} 25%, ${mid} 62%, ${light} 100%)` }} />;
};

/** "DISTORTED REALITY" + skull on a color smear, like Ben's video. `g` = global frame for beat sync. */
export const ShotTitle: React.FC<{ g: number; tint?: Tint; fadeOut?: boolean }> = ({ g, tint = "magenta", fadeOut }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const pulse = beatPulse(g + frame);
  const appear = interpolate(frame, [0, 6], [1.4, 1], { extrapolateRight: "clamp" });
  const shakeX = Math.sin(frame * 2.1) * 6 * pulse;
  const opacity = fadeOut ? interpolate(frame, [durationInFrames * 0.5, durationInFrames], [1, 0], { extrapolateLeft: "clamp" }) : 1;
  return (
    <AbsoluteFill style={{ opacity }}>
      <Smear tint={tint} />
      <div
        style={{
          position: "absolute",
          top: 60,
          width: "100%",
          textAlign: "center",
          fontFamily: "MetalMania",
          fontSize: 190,
          lineHeight: 0.95,
          color: "#120607",
          letterSpacing: 6,
          transform: `translateX(${shakeX}px) scale(${appear})`,
          textShadow: `0 0 ${18 + 30 * pulse}px rgba(0,0,0,0.6)`,
        }}
      >
        DISTORTED
        <br />
        REALITY
      </div>
      <Img
        src={staticFile("clip/skull.png")}
        style={{
          position: "absolute",
          left: 960 - 230,
          top: 470,
          width: 460,
          transform: `rotate(${-8 + 4 * Math.sin(frame / 20)}deg) scale(${1 + 0.08 * pulse})`,
          filter: "drop-shadow(0 20px 30px rgba(0,0,0,0.5))",
        }}
      />
    </AbsoluteFill>
  );
};

/** A ring of skulls turning around a center skull (Ben's kaleidoscope idea). */
export const ShotSkullMandala: React.FC<{ g: number; tint?: Tint }> = ({ g, tint = "red" }) => {
  const frame = useCurrentFrame();
  const pulse = beatPulse(g + frame);
  const [dark, mid] = TINTS[tint];
  const n = 8;
  return (
    <AbsoluteFill style={{ background: `radial-gradient(circle at 50% 50%, ${mid} 0%, ${dark} 70%)` }}>
      <AbsoluteFill style={{ transform: `rotate(${frame * 0.9}deg) scale(${1 + 0.06 * pulse})` }}>
        {Array.from({ length: n }).map((_, i) => (
          <Img
            key={i}
            src={staticFile("clip/skull.png")}
            style={{
              position: "absolute",
              left: 960 - 140,
              top: 540 - 190,
              width: 280,
              transformOrigin: "50% 50%",
              transform: `rotate(${(360 / n) * i}deg) translateY(-380px) ${i % 2 ? "scaleX(-1)" : ""}`,
            }}
          />
        ))}
      </AbsoluteFill>
      <Img
        src={staticFile("clip/skull.png")}
        style={{ position: "absolute", left: 960 - 170, top: 540 - 230, width: 340, transform: `rotate(${-frame * 1.4}deg) scale(${1 + 0.12 * pulse})` }}
      />
    </AbsoluteFill>
  );
};
