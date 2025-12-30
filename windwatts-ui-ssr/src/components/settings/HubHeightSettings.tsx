"use client";

import { useContext, useEffect, useMemo } from "react";
import { SettingsContext } from "../../providers/SettingsContext";
import { Box, Slider, Typography } from "@mui/material";
import { HUB_HEIGHTS } from "../../constants/hubSettings";

export function HubHeightSettings() {
  const { hubHeight, setHubHeight, preferredModel } = useContext(SettingsContext);

  const { values: availableHeights, interpolation: step } = useMemo(() => {
    if (preferredModel && HUB_HEIGHTS[preferredModel]) return HUB_HEIGHTS[preferredModel];
    return HUB_HEIGHTS.default;
  }, [preferredModel]);

  useEffect(() => {
    if (!availableHeights.includes(hubHeight)) {
      const closest = availableHeights.reduce((prev, curr) => Math.abs(curr - hubHeight) < Math.abs(prev - hubHeight) ? curr : prev);
      setHubHeight(closest);
    }
  }, [availableHeights, hubHeight, setHubHeight]);

  const hubHeightMarks = availableHeights.map((value: number) => ({ value, label: `${value}m` }));

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>Hub Height</Typography>
      <Typography variant="body1" gutterBottom>Choose a closest value (in meters) to the considered hub height:</Typography>
      <Slider
        value={hubHeight}
        onChange={(_, v) => typeof v === "number" && setHubHeight(v)}
        aria-labelledby="hub-height-slider"
        valueLabelDisplay="auto"
        step={step || 1}
        marks={hubHeightMarks}
        min={Math.min(...availableHeights)}
        max={Math.max(...availableHeights)}
      />
    </Box>
  );
}
