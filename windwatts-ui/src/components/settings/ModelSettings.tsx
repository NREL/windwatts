import { Box, ToggleButton, Typography } from "@mui/material";
import { SettingToggleButtonGroup } from "./SettingToggleButtonGroup";
import { useContext } from "react";
import { SettingsContext } from "../../providers/SettingsContext";
import { DataModel } from "../../types";

export function ModelSettings() {
  const { preferredModel, setPreferredModel } = useContext(SettingsContext);

  const handleModelChange = (
    _: React.MouseEvent<HTMLElement>,
    newModel: DataModel
  ) => {
    if (newModel !== null) {
      setPreferredModel(newModel);
    }
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        Preferred Model
      </Typography>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(12, 1fr)",
          gap: 1,
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Box sx={{ gridColumn: "span 4" }}>
          <Typography variant="body2">Model:</Typography>
        </Box>
        <Box
          sx={{
            gridColumn: "span 8",
            textAlign: "left",
          }}
        >
          <SettingToggleButtonGroup
            value={preferredModel}
            exclusive
            size="small"
            onChange={handleModelChange}
            aria-label="model"
          >
            <ToggleButton value="wtk">
              <Typography variant="body2">WTK</Typography>
            </ToggleButton>
            <ToggleButton value="era5">
              <Typography variant="body2">ERA5</Typography>
            </ToggleButton>
            <ToggleButton value="ensemble">
              <Typography variant="body2">Ensemble</Typography>
            </ToggleButton>
          </SettingToggleButtonGroup>
        </Box>
      </Box>
    </Box>
  );
}
