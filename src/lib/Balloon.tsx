import { useCurrentFrame } from "remotion";

/** A red balloon drifting up (SVG, city coordinates). */
export const Balloon: React.FC<{ x0?: number; y0?: number; speed?: number; scale?: number }> = ({
  x0 = 1250,
  y0 = 720,
  speed = 1.9,
  scale = 1,
}) => {
  const frame = useCurrentFrame();
  const x = x0 + frame * speed * 0.58 + Math.sin(frame / 18) * 14;
  const y = y0 - frame * speed;
  return (
    <g transform={`translate(${x} ${y}) rotate(${Math.sin(frame / 18) * 6}) scale(${scale})`}>
      <path d="M0 40 Q -6 70 4 100 Q 12 125 0 150" stroke="#141414" strokeWidth={2.5} fill="none" />
      <ellipse cx={0} cy={0} rx={30} ry={38} fill="#b3132e" stroke="#141414" strokeWidth={4} />
      <ellipse cx={-10} cy={-14} rx={7} ry={11} fill="#e8a0a8" opacity={0.7} />
      <path d="M-6 37 L6 37 L0 46 Z" fill="#b3132e" stroke="#141414" strokeWidth={3} />
    </g>
  );
};
