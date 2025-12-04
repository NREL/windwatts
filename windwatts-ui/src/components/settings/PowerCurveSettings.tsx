import { Box, FormControl, Typography, Select, MenuItem } from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import useSWR from "swr";
import { SettingsContext } from "../../providers/SettingsContext";
import { useContext } from "react";
import { getAvailablePowerCurves } from "../../services/api";
import { POWER_CURVE_LABEL } from "../../constants";

const DefaultPowerCurveOptions = [
  "nlr-reference-2.5kW",
  "nlr-reference-100kW",
  "nlr-reference-250kW",
  "nlr-reference-2000kW",
];

export function PowerCurveSettings() {
  const { powerCurve, setPowerCurve } = useContext(SettingsContext);

  // Fetch available power curves from the API
  const { data } = useSWR(
    "/api/wtk/available-powercurves",
    getAvailablePowerCurves,
    { fallbackData: { available_power_curves: DefaultPowerCurveOptions } }
  );

  const powerCurveOptions: string[] = data?.available_power_curves || [];

  const handlePowerCurveChange = (event: SelectChangeEvent<string>) => {
    setPowerCurve(event.target.value as string);
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        Power Curve
      </Typography>
      <Typography variant="body1" gutterBottom>
        Select a power curve option:
      </Typography>

      <FormControl component="fieldset" sx={{ width: "100%" }}>
        {powerCurveOptions.length > 0 ? (
          <>
            {/* <InputLabel id="power-curve-label">Power Curve</InputLabel> */}
            <Select
              labelId="power-curve-label"
              id="power-curve-select"
              value={powerCurve}
              // label="Power Curve"
              onChange={handlePowerCurveChange}
              fullWidth
              size="small"
            >
              {powerCurveOptions.map((option, idx) => (
                <MenuItem key={"power_curve_option_" + idx} value={option}>
                  {POWER_CURVE_LABEL[option] || option}
                </MenuItem>
              ))}
            </Select>
          </>
        ) : (
          <Typography variant="body2">
            Loading power curve options...
          </Typography>
        )}

        <Typography variant="body2" marginTop={2} gutterBottom>
          * Make sure the selected turbine class matches the hub height (higher
          hub heights should be chosen for larger turbines).
        </Typography>
      </FormControl>
    </Box>
  );
}
