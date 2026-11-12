import { Series } from "remotion";
import { Look } from "../lib/Look";
import { ShotCity, ShotDistorFront, ShotDistorSide, ShotEnd, ShotHatClose, ShotTwo } from "./pieces";

const S = 30; // frames per second

/** Test 2a: Distor front as a puppet (blink, nod, talk, tear). 8 s. */
export const Test2a: React.FC = () => (
  <Look>
    <ShotDistorFront />
  </Look>
);

/** Test 2b: two-shot, then close-ups (shot / reverse shot). 12 s. */
export const Test2b: React.FC = () => (
  <Look>
    <Series>
      <Series.Sequence durationInFrames={5 * S}>
        <ShotTwo talkFrom={30} talkTo={100} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={4 * S}>
        <ShotDistorSide facing="left" talkFrom={10} talkTo={75} blinks={[90]} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={3 * S}>
        <ShotHatClose />
      </Series.Sequence>
    </Series>
  </Look>
);

/** Test 2c: a first ~30 s sequence. */
export const TEST2C_FRAMES = (5 + 5 + 6 + 6 + 4 + 4) * S;
export const Test2c: React.FC = () => (
  <Look>
    <Series>
      <Series.Sequence durationInFrames={5 * S}>
        <ShotCity balloonY={1150} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={5 * S}>
        <ShotHatClose />
      </Series.Sequence>
      <Series.Sequence durationInFrames={6 * S}>
        <ShotDistorFront talkFrom={70} talkTo={140} blinks={[25, 150]} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={6 * S}>
        <ShotTwo talkFrom={40} talkTo={120} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={4 * S}>
        <ShotDistorSide facing="right" blinks={[40]} />
      </Series.Sequence>
      <Series.Sequence durationInFrames={4 * S}>
        <ShotEnd />
      </Series.Sequence>
    </Series>
  </Look>
);
