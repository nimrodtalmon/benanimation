import { Img, staticFile, useCurrentFrame } from "remotion";

/**
 * A cut-out character drawing. Children are drawn in the drawing's own pixel
 * coordinates (viewBox = drawing size), so overlays such as tears move with it.
 */
export const Character: React.FC<{
  name: string;
  size: [number, number]; // drawing size in pixels (public/chars/<name>.png)
  x: number;
  y: number;
  height: number; // on-screen height
  breathe?: number; // breathing amplitude in pixels
  boilId?: string;
  children?: React.ReactNode;
}> = ({ name, size, x, y, height, breathe = 6, boilId, children }) => {
  const frame = useCurrentFrame();
  const [w, h] = size;
  const width = (height * w) / h;
  const dy = Math.sin(frame / 22) * breathe;
  const tilt = Math.sin(frame / 37) * 0.35;
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y + dy,
        width,
        height,
        transform: `rotate(${tilt}deg)`,
        transformOrigin: "30% 100%",
        filter: boilId ? `url(#${boilId})` : undefined,
      }}
    >
      <Img src={staticFile(`chars/${name}.png`)} style={{ width: "100%", height: "100%" }} />
      <svg viewBox={`0 0 ${w} ${h}`} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
        {children}
      </svg>
    </div>
  );
};
