import { useContext } from "react";
import { Paper, Typography, FormControlLabel, Switch } from "@mui/material";
import { SettingsContext } from "../../providers/SettingsContext";

export const EnsembleSettings = () => {
  const { preferredModel, setPreferredModel } = useContext(SettingsContext);

  return (
    <>
      <Typography variant="h6" gutterBottom mt={2}>
        Experimental
      </Typography>
      <Paper sx={{ p: 2 }} variant="outlined">
        <FormControlLabel
          control={
            <Switch
              checked={preferredModel === "ensemble-quantiles"}
              onChange={(e) =>
                setPreferredModel(e.target.checked ? "ensemble-quantiles" : "era5-quantiles")
              }
            />
          }
          label="Enable Ensemble Model"
        />
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ mt: 1, fontStyle: "italic" }}
        >
          The WindWatts Ensemble model is an alternative to our default
          atmospheric model, ERA5. This new model leverages machine learning
          with data from multiple constituent models with ancillary location and
          terrain data. While early results show significant performance
          improvements, this model is still being developed and should be used
          with care.
        </Typography>
      </Paper>
    </>
  );
};
