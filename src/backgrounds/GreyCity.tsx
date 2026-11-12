import { AbsoluteFill, random, useCurrentFrame } from "remotion";

type Layer = { seed: string; base: number; minH: number; maxH: number; fill: string; window: string; parallax: number };

const LAYERS: Layer[] = [
  { seed: "far", base: 760, minH: 120, maxH: 360, fill: "#8a8b86", window: "#a9a58f", parallax: 0.15 },
  { seed: "mid", base: 880, minH: 180, maxH: 470, fill: "#5f6166", window: "#d8c27a", parallax: 0.35 },
  { seed: "near", base: 1080, minH: 260, maxH: 560, fill: "#3b3d43", window: "#e7c96a", parallax: 0.6 },
];

/** Jittered polygon point, so straight edges look hand-drawn. */
const j = (s: string, i: number, amt = 3) => (random(`${s}-${i}`) - 0.5) * 2 * amt;

const Buildings: React.FC<{ layer: Layer; shift: number }> = ({ layer, shift }) => {
  const out: React.ReactNode[] = [];
  let x = -120;
  let i = 0;
  while (x < 2200) {
    const r = (k: string) => random(`${layer.seed}-${i}-${k}`);
    const w = 90 + r("w") * 170;
    const h = layer.minH + r("h") * (layer.maxH - layer.minH);
    const top = layer.base - h;
    const roof = r("roof");
    const pts = [
      [x + j(layer.seed, i * 9), 1100],
      [x + j(layer.seed, i * 9 + 1), top + j(layer.seed, i * 9 + 2)],
      ...(roof > 0.7 ? [[x + w / 2, top - 40 - r("peak") * 40]] : []),
      [x + w + j(layer.seed, i * 9 + 3), top + j(layer.seed, i * 9 + 4)],
      [x + w + j(layer.seed, i * 9 + 5), 1100],
    ];
    out.push(
      <polygon key={`b${i}`} points={pts.map((p) => p.join(",")).join(" ")} fill={layer.fill} stroke="#141414" strokeWidth={4} strokeLinejoin="round" />,
    );
    // windows
    const cols = Math.floor(w / 34);
    const rows = Math.floor(h / 46);
    for (let c = 0; c < cols; c++) {
      for (let rr = 0; rr < rows - 1; rr++) {
        const lit = random(`${layer.seed}-${i}-${c}-${rr}`);
        if (lit < 0.72) continue;
        out.push(
          <rect
            key={`w${i}-${c}-${rr}`}
            x={x + 14 + c * 34}
            y={top + 22 + rr * 46}
            width={14}
            height={22}
            fill={layer.window}
            stroke="#141414"
            strokeWidth={2}
          />,
        );
      }
    }
    if (r("chimney") > 0.8) {
      out.push(
        <rect key={`c${i}`} x={x + w * 0.65} y={top - 110} width={26} height={112} fill={layer.fill} stroke="#141414" strokeWidth={4} />,
      );
    }
    x += w + 6 + r("gap") * 30;
    i++;
  }
  return <g transform={`translate(${-shift * layer.parallax} 0)`}>{out}</g>;
};

/** A grey city at dusk, drawn in code. `pan` slides the camera sideways (pixels). */
export const GreyCity: React.FC<{ pan?: number; children?: React.ReactNode }> = ({ pan = 0, children }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <svg viewBox="0 0 1920 1080" width="100%" height="100%">
        <defs>
          <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#b9b29a" />
            <stop offset="0.65" stopColor="#d7c9a0" />
            <stop offset="1" stopColor="#9c9784" />
          </linearGradient>
        </defs>
        <rect width="1920" height="1080" fill="url(#sky)" />
        <circle cx={1500 - pan * 0.05} cy={330} r={95} fill="#ece0b4" opacity={0.75} />
        {/* slow smog bands */}
        {[0, 1, 2].map((k) => (
          <rect key={k} x={-200 + ((frame * (0.3 + k * 0.15)) % 400)} y={420 + k * 90} width={2400} height={26} fill="#8f8a78" opacity={0.18} />
        ))}
        {children}
        {LAYERS.map((l) => (
          <Buildings key={l.seed} layer={l} shift={pan} />
        ))}
      </svg>
    </AbsoluteFill>
  );
};
