/** Rig math and shared drawing pieces. Everything is computed in world coordinates (1920 x 1080 stage). */

export type V = [number, number];
export const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1]];
export const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1]];
export const mul = (a: V, k: number): V => [a[0] * k, a[1] * k];
export const lerp = (a: V, b: V, u: number): V => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
export const len = (a: V) => Math.hypot(a[0], a[1]);
export const rot = (p: V, deg: number): V => {
  const r = (deg * Math.PI) / 180;
  return [p[0] * Math.cos(r) - p[1] * Math.sin(r), p[0] * Math.sin(r) + p[1] * Math.cos(r)];
};
/** World position of a point given in a part's local frame (origin `at`, rotated by `deg`). */
export const local = (at: V, deg: number, p: V): V => add(at, rot(p, deg));
/** Angle (deg) that makes a part drawn pointing down (+y) point from a to b. */
export const aim = (a: V, b: V) => (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI - 90;

/** Two-bone IK: elbow/knee position for a limb from `root` reaching `target`. bend = +1 / -1 picks the side. */
export const ik = (root: V, target: V, l1: number, l2: number, bend: 1 | -1): { mid: V; end: V } => {
  let d = sub(target, root);
  let dist = len(d);
  const max = (l1 + l2) * 0.999;
  if (dist > max) {
    d = mul(d, max / dist);
    dist = max;
  }
  const end = add(root, d);
  const a = Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist))));
  const base = Math.atan2(d[1], d[0]);
  const ang = base + bend * a;
  return { mid: [root[0] + l1 * Math.cos(ang), root[1] + l1 * Math.sin(ang)], end };
};

/**
 * IK for arms: of the two possible elbows, take the one that hangs lower and
 * further out to the arm's side (side = -1 screen-left arm, +1 screen-right arm),
 * so elbows never flip up or into the body.
 */
export const ikArm = (root: V, target: V, l1: number, l2: number, side: -1 | 1) => {
  const a = ik(root, target, l1, l2, 1);
  const b = ik(root, target, l1, l2, -1);
  const score = (m: V) => (m[1] - root[1]) + 0.6 * side * (m[0] - root[0]);
  return score(a.mid) >= score(b.mid) ? a : b;
};

export const INK = "#16120f";
export const ink = { stroke: INK, strokeWidth: 5, strokeLinejoin: "round" as const, strokeLinecap: "round" as const };

/** Place a part drawn in local coordinates (pivot at 0,0, pointing down) at a world point. */
export const At: React.FC<{ p: V; deg?: number; s?: number; children: React.ReactNode }> = ({ p, deg = 0, s = 1, children }) => (
  <g transform={`translate(${p[0]} ${p[1]}) rotate(${deg}) scale(${s})`}>{children}</g>
);

/**
 * A limb segment from a to b: tapered capsule with a cel shadow on one side.
 * w1/w2 = widths at the two ends.
 */
export const Seg: React.FC<{ a: V; b: V; w1: number; w2: number; fill: string; shade: string; shadeSide?: 1 | -1 }> = ({
  a,
  b,
  w1,
  w2,
  fill,
  shade,
  shadeSide = 1,
}) => {
  const L = len(sub(b, a));
  const deg = aim(a, b);
  const h1 = w1 / 2;
  const h2 = w2 / 2;
  const body = `M ${-h1} 0 L ${-h2} ${L} A ${h2} ${h2} 0 0 0 ${h2} ${L} L ${h1} 0 A ${h1} ${h1} 0 0 0 ${-h1} 0 Z`;
  const s = shadeSide;
  const shadow = `M ${s * h1 * 0.2} ${-h1 * 0.6} L ${s * h2 * 0.25} ${L + h2 * 0.7} L ${s * h2} ${L} L ${s * h1} 0 Z`;
  return (
    <At p={a} deg={deg}>
      <path d={body} fill={fill} />
      <path d={shadow} fill={shade} opacity={0.9} />
      <path d={body} fill="none" {...ink} />
    </At>
  );
};

/** A simple hand: round mitt; `grip` closes it around a stick/pick. */
export const Hand: React.FC<{ p: V; deg: number; r?: number; fill: string; shade: string; grip?: boolean }> = ({ p, deg, r = 17, fill, shade, grip }) => (
  <At p={p} deg={deg}>
    <ellipse cx={0} cy={r * 0.5} rx={r * 0.95} ry={r} fill={fill} {...ink} />
    <path d={`M ${r * 0.2} ${-r * 0.3} Q ${r * 0.95} ${r * 0.4} ${r * 0.3} ${r * 1.4}`} fill="none" stroke={shade} strokeWidth={r * 0.5} opacity={0.6} />
    {grip ? (
      <path d={`M ${-r * 0.7} ${r * 0.6} Q 0 ${r * 1.1} ${r * 0.7} ${r * 0.6}`} fill="none" {...ink} strokeWidth={3.5} />
    ) : (
      <path d={`M ${-r * 0.4} ${r * 1.3} L ${-r * 0.5} ${r * 1.8} M 0 ${r * 1.45} L 0 ${r * 2} M ${r * 0.4} ${r * 1.3} L ${r * 0.5} ${r * 1.8}`} {...ink} strokeWidth={4} fill="none" />
    )}
  </At>
);

/** Soft floor shadow under a character or prop. */
export const FloorShadow: React.FC<{ x: number; y: number; rx: number; ry?: number; o?: number }> = ({ x, y, rx, ry = rx * 0.16, o = 0.35 }) => (
  <ellipse cx={x} cy={y} rx={rx} ry={ry} fill="#000" opacity={o} filter="url(#softShadow)" />
);

/** Mouth substitution shapes (Hanna-Barbera / Rhubarb style). */
export type MouthShape = "rest" | "ah" | "oh" | "oo" | "ee";
export const Mouth: React.FC<{ shape: MouthShape; open: number; w?: number }> = ({ shape, open, w = 34 }) => {
  if (shape === "rest" || open < 0.08) return <path d={`M ${-w * 0.5} 0 Q 0 ${4} ${w * 0.5} -2`} fill="none" {...ink} strokeWidth={4.5} />;
  const o = 0.35 + 0.65 * open;
  const dims: Record<string, [number, number]> = { ah: [w * 0.55, w * 0.62], oh: [w * 0.38, w * 0.55], oo: [w * 0.24, w * 0.3], ee: [w * 0.62, w * 0.22] };
  const [rx, ry0] = dims[shape];
  const ry = ry0 * o;
  return (
    <g>
      <ellipse cx={0} cy={ry * 0.5} rx={rx} ry={ry} fill="#3a0f14" {...ink} strokeWidth={4.5} />
      {shape !== "oo" && <rect x={-rx * 0.6} y={ry * 0.5 - ry + 2} width={rx * 1.2} height={Math.min(9, ry * 0.4)} fill="#f4efe6" rx={2} />}
      {(shape === "ah" || shape === "oh") && <ellipse cx={0} cy={ry * 0.5 + ry * 0.55} rx={rx * 0.5} ry={ry * 0.3} fill="#c2404e" />}
    </g>
  );
};
