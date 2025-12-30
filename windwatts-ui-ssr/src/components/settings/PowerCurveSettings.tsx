"use client";

import { Box, FormControl, Typography, Select, MenuItem } from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { SettingsContext } from "../../providers/SettingsContext";
import { useContext } from "react";
import { POWER_CURVE_LABEL } from "../../constants/powerCurves";

export function PowerCurveSettings({ powerCurves }: { powerCurves: string[] }) {
  const { powerCurve, setPowerCurve } = useContext(SettingsContext);

  const handlePowerCurveChange = (event: SelectChangeEvent<string>) => {
    setPowerCurve(event.target.value as string);
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>Power Curve</Typography>
      <Typography variant="body1" gutterBottom>Select a power curve option:</Typography>
      <FormControl component="fieldset" sx={{ width: "100%" }}>
        {powerCurves.length > 0 ? (
          <Select value={powerCurve} onChange={handlePowerCurveChange} fullWidth size="small">
            {powerCurves.map((option, idx) => (
              <MenuItem key={`power_curve_option_${idx}`} value={option}>
                {POWER_CURVE_LABEL[option] || option}
              </MenuItem>
            ))}
          </Select>
        ) : (
          <Typography variant="body2">Loading power curve options...</Typography>
        )}
        <Typography variant="body2" marginTop={2} gutterBottom>
          * Make sure the selected turbine class matches the hub height (higher hub heights should be chosen for larger turbines).
        </Typography>
      </FormControl>
    </Box>
  );
}
