import { Composition } from "remotion";
import { Test1 } from "./shots/Test1";

export const FPS = 30;

export const Root: React.FC = () => (
  <>
    <Composition id="Test1" component={Test1} durationInFrames={10 * FPS} fps={FPS} width={1920} height={1080} />
  </>
);
