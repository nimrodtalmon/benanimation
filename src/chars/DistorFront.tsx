import { Img, staticFile } from "remotion";
import { Tear } from "../lib/Tear";

const W = 1150;
const H = 1600;
const PIVOT = [520, 1000]; // base of the neck
const EYES = [
  [305, 500, 535, 612],
  [620, 505, 855, 648],
];
const MOUTH = [562, 864];

/**
 * Ben's front drawing as a puppet: body and head are separate layers
 * (scripts/split_front.py); blinking eyelids, mouth and tear are drawn on top.
 */
export const DistorFront: React.FC<{
  x: number;
  y: number;
  height: number;
  blink?: number;
  mouth?: number;
  tilt?: number;
  nod?: number; // head moves down by this many drawing pixels
  tearDelay?: number;
}> = ({ x, y, height, blink = 0, mouth = 0, tilt = 0, nod = 0, tearDelay = 0 }) => {
  const width = (height * W) / H;
  const layer: React.CSSProperties = { position: "absolute", inset: 0, width: "100%", height: "100%" };
  return (
    <div style={{ position: "absolute", left: x, top: y, width, height }}>
      <Img src={staticFile("chars/distor-front-body.png")} style={layer} />
      <div
        style={{
          ...layer,
          transform: `translateY(${(nod / H) * 100}%) rotate(${tilt}deg)`,
          transformOrigin: `${(PIVOT[0] / W) * 100}% ${(PIVOT[1] / H) * 100}%`,
          filter: "url(#boil)",
        }}
      >
        <Img src={staticFile("chars/distor-front-head.png")} style={layer} />
        {/* eyelids: skin texture sliding down, limited to the eye whites */}
        <div
          style={{
            ...layer,
            WebkitMaskImage: `url(${staticFile("chars/distor-front-eyes.png")})`,
            WebkitMaskSize: "100% 100%",
            maskImage: `url(${staticFile("chars/distor-front-eyes.png")})`,
            maskSize: "100% 100%",
          }}
        >
          {blink > 0.02 &&
            EYES.map(([x0, y0, x1, y1], i) => (
              <div
                key={i}
                style={{
                  position: "absolute",
                  left: `${(x0 / W) * 100}%`,
                  top: `${(y0 / H) * 100}%`,
                  width: `${((x1 - x0) / W) * 100}%`,
                  height: `${(((y1 - y0) * Math.min(1, blink * 1.1)) / H) * 100}%`,
                  backgroundImage: `url(${staticFile("tex/skin.png")})`,
                  backgroundSize: "40% auto",
                  borderBottom: `${height * 0.008}px solid #151313`,
                }}
              />
            ))}
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={layer}>
          {mouth > 0.02 && (
            <ellipse
              cx={MOUTH[0]}
              cy={MOUTH[1] + 6 * mouth}
              rx={30 + 45 * mouth}
              ry={4 + 26 * mouth}
              fill="#3b1416"
              stroke="#151313"
              strokeWidth={6}
            />
          )}
          <Tear at={[345, 828]} run={110} cycle={90} delay={tearDelay} />
        </svg>
      </div>
    </div>
  );
};
