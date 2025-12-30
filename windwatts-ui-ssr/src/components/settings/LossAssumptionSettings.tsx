"use client";

import { useContext, useMemo, useState } from "react";
import { Box, TextField, Typography, InputAdornment, Switch, FormControlLabel } from "@mui/material";
import { SettingsContext } from "../../providers/SettingsContext";

export function LossAssumptionSettings() {
  const { lossAssumptionPercent, setLossAssumptionPercent } = useContext(SettingsContext);

  const [inputValue, setInputValue] = useState<string>(String(lossAssumptionPercent));

  const percent = useMemo(() => {
    const parsed = Number(inputValue);
    if (Number.isNaN(parsed)) return lossAssumptionPercent;
    return Math.max(0, Math.min(100, Math.round(parsed)));
  }, [inputValue, lossAssumptionPercent]);

  const handleCommit = (value: number) => {
    setLossAssumptionPercent(value);
    setInputValue(String(value));
  };

  const enabled = percent > 0;

  const handleToggle = (on: boolean) => {
    if (on) handleCommit(17);
    else handleCommit(0);
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>Loss Assumption</Typography>
      <Typography variant="body2" mb={2}>Set an energy percent loss (17% recommended).</Typography>
      <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
        <FormControlLabel sx={{ flexShrink: 0, mr: 1 }} control={<Switch checked={enabled} onChange={(e) => handleToggle(e.target.checked)} />} label="Enable" />
        <TextField
          size="small"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onBlur={() => handleCommit(percent)}
          inputProps={{ inputMode: "numeric", pattern: "[0-9]*", min: 0, max: 100 }}
          fullWidth
          sx={{ flex: 1, minWidth: 0 }}
          label="Loss"
          disabled={!enabled}
          InputProps={{ endAdornment: <InputAdornment position="end">%</InputAdornment> }}
        />
      </Box>
    </Box>
  );
}
