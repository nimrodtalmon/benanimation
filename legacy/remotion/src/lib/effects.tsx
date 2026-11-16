import { AbsoluteFill, useCurrentFrame } from "remotion";

/** Hand-drawn "line boil": outlines wobble slightly, redrawn every `every` frames. */
export const Boil: React.FC<{ id: string; strength?: number; every?: number }> = ({ id, strength = 2.5, every = 4 }) => {
  const frame = useCurrentFrame();
  const seed = Math.floor(frame / every) % 7;
  return (
    <svg width={0} height={0} style={{ position: "absolute" }}>
      <filter id={id}>
        <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves={2} seed={seed} />
        <feDisplacementMap in="SourceGraphic" scale={strength} xChannelSelector="R" yChannelSelector="G" />
      </filter>
    </svg>
  );
};

/** Film grain over the whole frame. */
export const Grain: React.FC<{ opacity?: number }> = ({ opacity = 0.14 }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ mixBlendMode: "overlay", opacity, pointerEvents: "none" }}>
      <svg width="100%" height="100%">
        <filter id="grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={2} seed={frame % 50} />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#grain)" />
      </svg>
    </AbsoluteFill>
  );
};

export const Vignette: React.FC<{ strength?: number }> = ({ strength = 0.55 }) => (
  <AbsoluteFill
    style={{
      pointerEvents: "none",
      background: `radial-gradient(ellipse at 55% 50%, rgba(0,0,0,0) 45%, rgba(0,0,0,${strength}) 100%)`,
    }}
  />
);
