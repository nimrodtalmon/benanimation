import { Composition } from "remotion";
import { SidePreview } from "./shots/SidePreview";
import { Test1 } from "./shots/Test1";
import { TEST2C_FRAMES, Test2a, Test2b, Test2c } from "./shots/Test2";

export const FPS = 30;
const HD = { fps: FPS, width: 1920, height: 1080 };

export const Root: React.FC = () => (
  <>
    <Composition id="Test1" component={Test1} durationInFrames={10 * FPS} {...HD} />
    <Composition id="Test2a" component={Test2a} durationInFrames={8 * FPS} {...HD} />
    <Composition id="Test2b" component={Test2b} durationInFrames={12 * FPS} {...HD} />
    <Composition id="Test2c" component={Test2c} durationInFrames={TEST2C_FRAMES} {...HD} />
    <Composition id="SidePreview" component={SidePreview} durationInFrames={1} fps={FPS} width={1200} height={1600} />
  </>
);
