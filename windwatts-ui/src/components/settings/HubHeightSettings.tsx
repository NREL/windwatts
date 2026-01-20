import { useContext, useEffect, useMemo } from "react";
import { SettingsContext } from "../../providers/SettingsContext";
import { Box, Slider, Typography, Paper } from "@mui/material";
import { HUB_HEIGHTS, TURBINE_DATA, TurbineInfo } from "../../constants";

export function HubHeightSettings() {
  const {
    hubHeight,
    setHubHeight,
    preferredModel: dataModel,
    powerCurve,
  } = useContext(SettingsContext);

  const { values: availableHeights, interpolation: step } = useMemo(() => {
    if (dataModel && HUB_HEIGHTS[dataModel]) {
      return HUB_HEIGHTS[dataModel];
    }
    return HUB_HEIGHTS.default;
  }, [dataModel]);

  // ensure slider compatibility with model switching and available heights changes
  useEffect(() => {
    if (!availableHeights.includes(hubHeight)) {
      // if current height not available, set to the closest available height
      const closestHeight = availableHeights.reduce((prev, curr) =>
        Math.abs(curr - hubHeight) < Math.abs(prev - hubHeight) ? curr : prev
      );
      setHubHeight(closestHeight);
    }
  }, [availableHeights, hubHeight, setHubHeight]);

  const hubHeightMarks = availableHeights.map((value: number) => ({
    value: value,
    label: `${value}m`,
  }));

  const handleHubHeightChange = (
    _: Event,
    newHubHeight: number | number[] | null
  ) => {
    if (newHubHeight !== null && typeof newHubHeight === "number") {
      setHubHeight(newHubHeight);
    }
  };

  const turbineData: TurbineInfo | undefined = TURBINE_DATA[powerCurve];

  const isHeightInRange: boolean = turbineData
    ? hubHeight >= turbineData.minHeight && hubHeight <= turbineData.maxHeight
    : true;

  const validationColor: "primary" | "success" | "error" = turbineData
    ? isHeightInRange
      ? "success"
      : "error"
    : "primary";

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Hub Height
      </Typography>
      <Typography variant="body1" gutterBottom>
        Choose a closest value (in meters) to the considered hub height:
      </Typography>

      {turbineData && (
        <Paper
          sx={{
            p: 1.5,
            mb: 2,
            backgroundColor: `${validationColor}.light`,
            borderLeft: `4px solid`,
            borderLeftColor: `${validationColor}.main`,
          }}
        >
          <Typography variant="body2">
            <strong>
              {isHeightInRange ? "Within" : "Outside"} recommended range - (
              {turbineData.minHeight}m - {turbineData.maxHeight}m)
            </strong>
          </Typography>
        </Paper>
      )}

      <Slider
        value={hubHeight}
        onChange={handleHubHeightChange}
        aria-labelledby="hub-height-slider"
        valueLabelDisplay="auto"
        getAriaValueText={(value) => `${value}m`}
        step={step}
        marks={hubHeightMarks}
        min={Math.min(...availableHeights)}
        max={Math.max(...availableHeights)}
        color={validationColor}
      />
    </Box>
  );
}
