import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { GreyCity } from "../backgrounds/GreyCity";
import { DistorFront } from "../chars/DistorFront";
import { DistorSide } from "../chars/DistorSide";
import { blinkAt, sway, talk } from "../lib/anim";
import { Balloon } from "../lib/Balloon";
import { Character } from "../lib/Character";
import { Tear } from "../lib/Tear";

/** Shots used by the tests. Each uses its own local frame (0 = start of shot). */

const HAT = { name: "hat-profile", size: [1451, 2023] as [number, number], tearEnd: [807, 1085] as [number, number] };

const useProgress = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  return { frame, p: frame / durationInFrames };
};

/** Camera: slow push-in around a point (percent of frame). */
const Camera: React.FC<{ from?: number; to?: number; origin?: string; children: React.ReactNode }> = ({
  from = 1,
  to = 1.06,
  origin = "50% 45%",
  children,
}) => {
  const { p } = useProgress();
  return <AbsoluteFill style={{ transform: `scale(${interpolate(p, [0, 1], [from, to])})`, transformOrigin: origin }}>{children}</AbsoluteFill>;
};

const City: React.FC<{ pan?: number; soft?: boolean; balloon?: boolean; balloonY?: number }> = ({ pan = 0, soft, balloon, balloonY }) => (
  <AbsoluteFill style={{ filter: soft ? "url(#boilBg) blur(3px) brightness(0.88)" : "url(#boilBg)", transform: "scale(1.03)" }}>
    <GreyCity pan={pan}>{balloon && <Balloon y0={balloonY} />}</GreyCity>
  </AbsoluteFill>
);

/** Wide city, camera pans right, balloon rises. */
export const ShotCity: React.FC<{ balloonY?: number }> = ({ balloonY = 900 }) => {
  const { p } = useProgress();
  return (
    <Camera from={1.04} to={1.0}>
      <City pan={interpolate(p, [0, 1], [-120, 220])} balloon balloonY={balloonY} />
    </Camera>
  );
};

/** Close-up of the hat character (Ben's drawing), tear running. */
export const ShotHatClose: React.FC = () => {
  const { p } = useProgress();
  return (
    <Camera to={1.08} origin="22% 38%">
      <City pan={interpolate(p, [0, 1], [0, 160])} />
      <Character name={HAT.name} size={HAT.size} x={-30} y={30} height={1090} boilId="boil">
        <Tear at={HAT.tearEnd} run={150} cycle={80} delay={10} />
      </Character>
    </Camera>
  );
};

/** Distor from the front (Ben's drawing as a puppet): blinks, nods, talks. */
type MouthFn = (frame: number) => number;

export const ShotDistorFront: React.FC<{ talkFrom?: number; talkTo?: number; blinks?: number[]; mouthFn?: MouthFn }> = ({
  talkFrom = 50,
  talkTo = 120,
  blinks = [30, 135, 200],
  mouthFn,
}) => {
  const { frame } = useProgress();
  return (
    <Camera to={1.07} origin="25% 40%">
      <City soft pan={40} />
      <DistorFront
        x={-110}
        y={-10}
        height={1500}
        blink={blinkAt(frame, blinks)}
        mouth={mouthFn ? mouthFn(frame) : talk(frame, talkFrom, talkTo)}
        tilt={sway(frame, 1.6, 140)}
        nod={Math.max(0, sway(frame, 10, 90))}
      />
    </Camera>
  );
};

/** Distor side view, close: `facing` right shows the tear side. */
export const ShotDistorSide: React.FC<{ facing?: "left" | "right"; talkFrom?: number; talkTo?: number; blinks?: number[]; mouthFn?: MouthFn }> = ({
  facing = "right",
  talkFrom = -1,
  talkTo = -1,
  blinks = [25, 110],
  mouthFn,
}) => {
  const { frame } = useProgress();
  const h = 1700;
  const w = (h * 1200) / 1600;
  const x = facing === "right" ? 700 - w / 2 : 1220 - w / 2;
  return (
    <Camera to={1.06} origin={facing === "right" ? "40% 40%" : "60% 40%"}>
      <City soft pan={facing === "right" ? 80 : -60} />
      <div style={{ position: "absolute", left: x, top: -130, width: w, height: h }}>
        <DistorSide uid="side" facing={facing} blink={blinkAt(frame, blinks)} mouth={mouthFn ? mouthFn(frame) : talk(frame, talkFrom, talkTo)} tilt={sway(frame, 1.5, 120)} />
      </div>
    </Camera>
  );
};

/** Two-shot: hat character (left, facing right) and Distor (right, facing left). */
export const ShotTwo: React.FC<{ talkFrom?: number; talkTo?: number; mouthFn?: MouthFn }> = ({ talkFrom = 40, talkTo = 110, mouthFn }) => {
  const { frame, p } = useProgress();
  const h = 1250;
  const w = (h * 1200) / 1600;
  return (
    <Camera from={1.0} to={1.04} origin="50% 40%">
      <City pan={interpolate(p, [0, 1], [60, 120])} balloon balloonY={1000} />
      <Character name={HAT.name} size={HAT.size} x={-30} y={170} height={950} boilId="boil">
        <Tear at={HAT.tearEnd} run={150} cycle={80} delay={25} />
      </Character>
      <div style={{ position: "absolute", left: 1920 - w + 120, top: 60, width: w, height: h }}>
        <DistorSide uid="two" facing="left" blink={blinkAt(frame, [70, 150])} mouth={mouthFn ? mouthFn(frame) : talk(frame, talkFrom, talkTo)} tilt={sway(frame, 1.2, 110, 1)} />
      </div>
    </Camera>
  );
};

/** End: city, balloon high up, fade to black. */
export const ShotEnd: React.FC = () => {
  const { p } = useProgress();
  return (
    <AbsoluteFill style={{ opacity: interpolate(p, [0.55, 1], [1, 0], { extrapolateLeft: "clamp" }) }}>
      <Camera from={1.0} to={1.05} origin="70% 20%">
        <City pan={260} balloon balloonY={500} />
      </Camera>
    </AbsoluteFill>
  );
};
