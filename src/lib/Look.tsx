import { AbsoluteFill } from "remotion";
import { Boil, Grain, Vignette } from "./effects";

/** The shared look of every shot: line boil filters, color grade, vignette, grain. */
export const Look: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill style={{ backgroundColor: "#111", filter: "contrast(1.06) saturate(0.92) sepia(0.08)" }}>
    <Boil id="boil" strength={2.2} />
    <Boil id="boilBg" strength={3} every={5} />
    {children}
    <Vignette />
    <Grain />
  </AbsoluteFill>
);
