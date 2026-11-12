import { AbsoluteFill } from "remotion";
import { DistorSide } from "../chars/DistorSide";

export const SidePreview: React.FC = () => (
  <AbsoluteFill style={{ backgroundColor: "#ece6da" }}>
    <div style={{ position: "absolute", left: 0, top: 0, width: 1200, height: 1600 }}>
      <DistorSide uid="p" facing="right" />
    </div>
  </AbsoluteFill>
);
