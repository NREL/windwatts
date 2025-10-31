"use client";

import { Box, Typography, IconButton } from "@mui/material";
import { Close } from "@mui/icons-material";
import { UnitsSettings } from "./UnitsSettings";
import { useContext, useState } from "react";
import { SettingsContext } from "../../providers/SettingsContext";
import { PowerCurveSettings } from "./PowerCurveSettings";
import { LossAssumptionSettings } from "./LossAssumptionSettings";
import { HubHeightSettings } from "./HubHeightSettings";
import { EnsembleSettings } from "./EnsembleSettings";

export const Settings = ({ powerCurves }: { powerCurves: string[] }) => {
  const { toggleSettings } = useContext(SettingsContext);
  const [hasScrolled, setHasScrolled] = useState(false);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setHasScrolled(e.currentTarget.scrollTop > 0);
  };

  return (
    <Box sx={{ position: "relative", width: "100%", bgcolor: "background.paper", maxHeight: "80vh", display: "flex", flexDirection: "column", borderRadius: 2, overflow: "hidden" }}>
      {/* Sticky Header */}
      <Box sx={{ position: "sticky", top: 0, bgcolor: "background.paper", p: 3, pb: 2, borderBottom: hasScrolled ? "1px solid rgba(0,0,0,0.12)" : "1px solid transparent", boxShadow: hasScrolled ? "0 2px 8px rgba(0,0,0,0.1)" : "none", transition: "all 0.2s ease", zIndex: 1 }}>
        <IconButton aria-label="close" onClick={toggleSettings} sx={{ position: "absolute", right: 8, top: 8 }}>
          <Close />
        </IconButton>
        <Typography variant="h5" component="h2">Settings</Typography>
      </Box>
      {/* Scrollable Content */}
      <Box sx={{ flex: 1, overflowY: "auto", p: 3, pt: 1 }} onScroll={handleScroll}>
        <HubHeightSettings />
        <PowerCurveSettings powerCurves={powerCurves} />
        <UnitsSettings />
        <LossAssumptionSettings />
        <EnsembleSettings />
      </Box>
    </Box>
  );
};
