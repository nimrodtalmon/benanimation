import { AbsoluteFill, Audio, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Grain, Vignette } from "../lib/effects";
import { SONG, SPB, decay, onTwos, ring } from "../band/song";

/**
 * Test: one real painting (Pepper, by David Revoy, CC BY 4.0) as a puppet.
 * Head and body are separate layers; blinks and mouth shapes are painted on top,
 * driven by the same song data as the band (mouth shapes from Rhubarb on the vocal track).
 * Drawing coordinates: 1786 x 2207 (the painting).
 */
export const PAINT_START = 8; // seconds into the song
export const PAINT_FRAMES = 12 * 30;

const W = 1786;
const H = 2207;
const PIVOT = [900, 1090];
const SKIN = "rgb(186,178,143)";
const EYES = [
  { cx: 790, cy: 750, rx: 60, ry: 40 },
  { cx: 1012, cy: 748, rx: 58, ry: 40 },
];
const MOUTH = [910, 922];

const mouthAt = (t: number) => {
  for (const n of SONG.vocal) {
    if (t < n.t - 0.03 || t > n.t + n.dur + 0.04) continue;
    const a = Math.min(1, (t - n.t + 0.03) / 0.06);
    const r = Math.min(1, (n.t + n.dur + 0.04 - t) / 0.06);
    return { open: Math.min(a, r) * ((n as { open?: number }).open ?? 1), vowel: n.vowel };
  }
  return { open: 0, vowel: "rest" };
};

const blinkAt = (t: number) => {
  const k = Math.floor(t / 3.1);
  const s = k * 3.1 + ((k * 7) % 10) / 12;
  const d = t - s;
  return d >= 0 && d < 0.14 ? 1 - Math.abs(d - 0.07) / 0.07 : 0;
};

export const PaintTest: React.FC = () => {
  const frame = useCurrentFrame();
  const t = PAINT_START + onTwos(frame);
  const tc = PAINT_START + frame / 30;
  const phase = (t % SPB) / SPB;
  const onBeat = Math.pow(Math.cos(Math.PI * phase), 2);
  const beats = Array.from({ length: 200 }, (_, i) => i * SPB);
  const tilt = Math.sin(t * 0.9) * 2.2 + ring(t, beats, 0.2, 2, 0.8);
  const nod = onBeat * 9;
  const m = mouthAt(t);
  const blink = blinkAt(t);
  const s = 1500 / H;
  const zoom = interpolate(frame, [0, PAINT_FRAMES], [1, 1.07]);
  const flash = decay(tc, beats, 0.15);

  const wide = m.vowel === "ee" ? 1.25 : m.vowel === "oo" ? 0.6 : m.vowel === "oh" ? 0.8 : 1;
  const mh = 6 + 30 * m.open * (m.vowel === "ee" ? 0.5 : 1);
  const mw = (26 + 14 * m.open) * wide;

  return (
    <AbsoluteFill style={{ backgroundColor: "#111" }}>
      <Audio src={staticFile("song/song.wav")} startFrom={PAINT_START * 30} />
      <AbsoluteFill style={{ transform: `scale(${zoom})`, transformOrigin: "50% 45%" }}>
        <Img src={staticFile("paint/pepper-bg.jpg")} style={{ position: "absolute", width: 1920, height: 1080, objectFit: "cover", filter: `brightness(${0.75 + 0.2 * flash})` }} />
        {[300, 1600, 520, 1420].map((x, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left: x - 140,
              top: 140 + (i % 2) * 380,
              width: 280,
              height: 280,
              borderRadius: "50%",
              background: ["#7fd6c2", "#e7b46a", "#a58be0", "#e07a8b"][i],
              opacity: 0.18 + 0.25 * (Math.floor(tc / (SPB * 4)) % 4 === i ? flash : 0),
              filter: "blur(40px)",
            }}
          />
        ))}
        <div style={{ position: "absolute", left: (1920 - W * s) / 2, top: -30 + Math.sin(tc * 1.6) * 3, width: W * s, height: H * s }}>
          <Img src={staticFile("paint/pepper-body.png")} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
          <div
            style={{
              position: "absolute",
              inset: 0,
              transform: `translateY(${nod * s}px) rotate(${tilt}deg)`,
              transformOrigin: `${(PIVOT[0] / W) * 100}% ${(PIVOT[1] / H) * 100}%`,
            }}
          >
            <Img src={staticFile("paint/pepper-head.png")} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
            <svg viewBox={`0 0 ${W} ${H}`} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
              <defs>
                <filter id="feather">
                  <feGaussianBlur stdDeviation={5} />
                </filter>
                <filter id="soft">
                  <feGaussianBlur stdDeviation={1.6} />
                </filter>
                {EYES.map((e, i) => (
                  <mask key={i} id={`eye${i}`} maskUnits="userSpaceOnUse" x={0} y={0} width={W} height={H}>
                    <ellipse cx={e.cx} cy={e.cy - 4} rx={e.rx + 12} ry={e.ry + 16} fill="#fff" filter="url(#feather)" />
                  </mask>
                ))}
              </defs>
              {/* blink: real painted skin (taken from just below the eye) slides down, lash line at its edge */}
              {blink > 0.02 &&
                EYES.map((e, i) => {
                  const y = e.cy - e.ry + 2 * e.ry * blink;
                  return (
                    <g key={i}>
                      <clipPath id={`lid${i}`}>
                        <rect x={e.cx - e.rx - 20} y={e.cy - e.ry - 30} width={2 * e.rx + 40} height={y - (e.cy - e.ry) + 30} />
                      </clipPath>
                      <g mask={`url(#eye${i})`}>
                        <g clipPath={`url(#lid${i})`}>
                          <image href={staticFile("paint/pepper-head.png")} x={0} y={-118} width={W} height={H} />
                        </g>
                      </g>
                      <path filter="url(#soft)" d={`M ${e.cx - e.rx + 4} ${y - 6} Q ${e.cx} ${y + 6 + 8 * blink} ${e.cx + e.rx - 4} ${y - 8}`} stroke="#2a1d1f" strokeWidth={8} fill="none" strokeLinecap="round" />
                    </g>
                  );
                })}
              {/* mouth: painted-looking open shape over the closed one */}
              {m.open > 0.05 && (
                <g filter="url(#soft)">
                  <ellipse cx={MOUTH[0]} cy={MOUTH[1] + mh * 0.35} rx={mw + 4} ry={mh * 0.5 + 4} fill="rgb(150,120,100)" opacity={0.6} />
                  <ellipse cx={MOUTH[0]} cy={MOUTH[1] + mh * 0.35} rx={mw} ry={mh * 0.5} fill="#3b1f22" />
                  <ellipse cx={MOUTH[0]} cy={MOUTH[1] + mh * 0.62} rx={mw * 0.55} ry={mh * 0.18} fill="#8a4a4a" opacity={0.8} />
                  {m.vowel !== "oo" && <rect x={MOUTH[0] - mw * 0.6} y={MOUTH[1] - mh * 0.12} width={mw * 1.2} height={Math.min(8, mh * 0.25)} fill="#e8e2d2" opacity={0.85} rx={3} />}
                </g>
              )}
            </svg>
          </div>
        </div>
      </AbsoluteFill>
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 26 }}>
        <div style={{ color: "#e9e1cf", fontFamily: "sans-serif", fontSize: 20, textShadow: "0 2px 6px #000", opacity: 0.85 }}>
          Painting: "Portrait of Pepper" by David Revoy, CC BY 4.0 (animated, cut into parts) · {(SONG as unknown as { credit: string }).credit}
        </div>
      </AbsoluteFill>
      <Vignette strength={0.45} />
      <Grain opacity={0.08} />
    </AbsoluteFill>
  );
};
