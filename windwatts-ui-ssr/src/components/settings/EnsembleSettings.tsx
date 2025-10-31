"use client";

import { useContext } from "react";
import { Paper, Typography, FormControlLabel, Switch } from "@mui/material";
import { SettingsContext } from "../../providers/SettingsContext";

export const EnsembleSettings = () => {
  const { ensemble, setEnsemble } = useContext(SettingsContext);
  return (
    <>
      <Typography variant="subtitle1" gutterBottom>Experimental</Typography>
      <Paper sx={{ p: 2 }} variant="outlined">
        <FormControlLabel control={<Switch checked={!!ensemble} onChange={(e) => setEnsemble(e.target.checked)} />} label="Enable Ensemble Model" />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, fontStyle: "italic" }}>
          The WindWatts Ensemble model is an alternative to our default atmospheric model, ERA5. This new model leverages machine learning with data from multiple constituent models with ancillary location and terrain data. While early results show significant performance improvements, this model is still being developed and should be used with care.
        </Typography>
      </Paper>
    </>
  );
};
