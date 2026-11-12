import { AbsoluteFill, Audio, Sequence, continueRender, delayRender, staticFile, useCurrentFrame } from "remotion";
import { Look } from "../lib/Look";
import { ShotCity, ShotDistorFront, ShotDistorSide, ShotHatClose, ShotTwo } from "../shots/pieces";
import { ShotSkullMandala, ShotTitle, TINTS, Tint } from "./extraShots";
import { SONG_FRAMES, beatPulse, lastBeat, onsetAt, singing, snap } from "./music";

// Title font (Metal Mania, OFL) — wait for it before rendering.
const fontHandle = delayRender("font");
new FontFace("MetalMania", `url(${staticFile("fonts/metal-mania-latin-400-normal.woff2")})`)
  .load()
  .then((f) => {
    (document.fonts as unknown as { add: (f: FontFace) => void }).add(f);
    continueRender(fontHandle);
  })
  .catch(() => continueRender(fontHandle));

type ShotKind = "city" | "hat" | "front" | "two" | "sideR" | "sideL" | "title" | "mandala";
type Cut = { at: number; shot: ShotKind; tint?: Tint };

/** The edit, in song seconds (snapped to the nearest beat). Sections: intro 0, B 15.5, C 32.4, D 44.5, E 66.8, outro 85.8. */
const EDIT: Cut[] = [
  // intro: slow
  { at: 0, shot: "city" },
  { at: 7.6, shot: "hat" },
  // B: the band comes in
  { at: 15.5, shot: "title", tint: "magenta" },
  { at: 19.2, shot: "front" },
  { at: 23.0, shot: "hat", tint: "green" },
  { at: 26.7, shot: "two" },
  { at: 30.4, shot: "mandala", tint: "red" },
  // C: calmer
  { at: 32.4, shot: "sideR", tint: "green" },
  { at: 38.0, shot: "city", tint: "green" },
  { at: 41.0, shot: "front" },
  // D: fast cuts, one per bar
  ...(() => {
    const order: [ShotKind, Tint | undefined][] = [
      ["mandala", "magenta"], ["front", "magenta"], ["hat", "green"], ["title", "green"], ["two", "blue"], ["sideL", undefined],
      ["mandala", "blue"], ["front", "green"], ["hat", "magenta"], ["sideR", "blue"], ["title", "blue"], ["two", undefined],
    ];
    const bar = (4 * 60) / 129.2;
    const cuts: Cut[] = [];
    for (let t = 44.5, i = 0; t < 66.5; t += bar, i++) cuts.push({ at: t, shot: order[i % order.length][0], tint: order[i % order.length][1] });
    return cuts;
  })(),
  // E: last round
  { at: 66.8, shot: "two", tint: "blue" },
  { at: 70.5, shot: "front" },
  { at: 74.2, shot: "hat", tint: "magenta" },
  { at: 77.9, shot: "sideR" },
  { at: 81.6, shot: "mandala", tint: "red" },
  // outro
  { at: 85.8, shot: "title", tint: "magenta" },
];

const Shot: React.FC<{ cut: Cut; g: number }> = ({ cut, g }) => {
  const sing = (f: number) => singing(g + f);
  switch (cut.shot) {
    case "city":
      return <ShotCity balloonY={1000} />;
    case "hat":
      return <ShotHatClose />;
    case "front":
      return <ShotDistorFront mouthFn={sing} blinks={[20, 95]} />;
    case "two":
      return <ShotTwo mouthFn={sing} />;
    case "sideR":
      return <ShotDistorSide facing="right" mouthFn={sing} blinks={[30, 120]} />;
    case "sideL":
      return <ShotDistorSide facing="left" mouthFn={sing} blinks={[25]} />;
    case "title":
      return <ShotTitle g={g} tint={cut.tint} fadeOut={g > 85 * 30} />;
    case "mandala":
      return <ShotSkullMandala g={g} tint={cut.tint} />;
  }
};

/** Color wash over character shots, like the color filters in Ben's video. */
const Wash: React.FC<{ tint?: Tint; g: number }> = ({ tint, g }) => {
  const frame = useCurrentFrame();
  if (!tint) return null;
  const pulse = beatPulse(g + frame, 10);
  return <AbsoluteFill style={{ backgroundColor: TINTS[tint][1], mixBlendMode: "color", opacity: 0.35 + 0.25 * pulse, pointerEvents: "none" }} />;
};

/** Song-wide effects: beat bump (from the band's entry on), invert flashes in the fast part. */
const FX: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  const sec = frame / 30;
  const bump = sec > 15.4 && sec < 86 ? beatPulse(frame, 7) : 0;
  const { index, since } = lastBeat(frame);
  const fast = sec > 44.4 && sec < 66.8;
  const invert = fast && index % 8 === 0 && since < 2;
  const shake = fast && onsetAt(frame) > 0.8 ? Math.sin(frame * 3.1) * 8 : 0;
  return (
    <AbsoluteFill
      style={{
        transform: `scale(${1 + 0.025 * bump}) translateX(${shake}px)`,
        filter: invert ? "invert(1) hue-rotate(180deg)" : undefined,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

export const Test3: React.FC = () => {
  const starts = EDIT.map((c) => snap(c.at));
  starts[0] = 0;
  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <Audio src={staticFile("clip/distorted-reality.m4a")} />
      <FX>
        <Look>
          {EDIT.map((cut, i) => {
            const from = starts[i];
            const to = i + 1 < EDIT.length ? starts[i + 1] : SONG_FRAMES;
            if (to <= from) return null;
            return (
              <Sequence key={i} from={from} durationInFrames={to - from}>
                <Shot cut={cut} g={from} />
                {cut.shot !== "title" && cut.shot !== "mandala" && <Wash tint={cut.tint} g={from} />}
              </Sequence>
            );
          })}
        </Look>
      </FX>
    </AbsoluteFill>
  );
};
