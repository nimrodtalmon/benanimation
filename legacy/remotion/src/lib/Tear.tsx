import { interpolate, useCurrentFrame } from "remotion";

/** A drop that swells at `at`, runs down `run` pixels, and fades. Repeats every `cycle` frames. */
export const Tear: React.FC<{ at: [number, number]; run?: number; cycle?: number; delay?: number; color?: string }> = ({
  at,
  run = 150,
  cycle = 80,
  delay = 0,
  color = "#b3132e",
}) => {
  const frame = useCurrentFrame();
  const t = (((frame - delay) % cycle) + cycle) % cycle;
  const swell = interpolate(t, [0, 30], [0, 1], { extrapolateRight: "clamp" });
  const slide = interpolate(t, [30, 62], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const fade = interpolate(t, [62, 78], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const ease = slide * slide; // gravity
  const r = 13 * swell;
  const [x, y0] = at;
  const y = y0 + ease * run;
  const tail = 10 + ease * run * 0.6; // streak left behind
  return (
    <g opacity={fade}>
      <rect x={x - r * 0.45} y={y - tail} width={r * 0.9} height={tail} fill={color} opacity={0.85} rx={r * 0.45} />
      <path d={`M ${x} ${y - r * 2.2} Q ${x + r} ${y - r * 0.2} ${x} ${y + r} Q ${x - r} ${y - r * 0.2} ${x} ${y - r * 2.2} Z`} fill={color} />
    </g>
  );
};
