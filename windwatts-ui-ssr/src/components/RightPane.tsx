"use client";

import { Box, Typography, Paper, Grid, Divider, Button } from "@mui/material";
import { styled } from "@mui/material/styles";
import SettingsIcon from "@mui/icons-material/Settings";
import { useContext, useState } from "react";
import { SettingsContext } from "../providers/SettingsContext";
import { POWER_CURVE_LABEL } from "../constants/powerCurves";
import AnalysisResults from "./resultPane/AnalysisResults";
import SummaryBar from "./SummaryBar";
import Controls from "./Controls";

const Item = styled(Paper)(({ theme }) => ({
  ...theme.typography.body2,
  padding: theme.spacing(1),
  textAlign: "center",
  color: theme.palette.text.secondary,
}));

export default function RightPane({ powerCurves, wind, production, error }: { powerCurves: string[]; wind: any; production: any; error?: string | null }) {
  const { currentPosition, hubHeight, powerCurve, toggleSettings } = useContext(SettingsContext);
  const { lat, lng } = currentPosition ?? {};

  const settingOptions = [
    { title: "Location", data: currentPosition && lat && lng ? `${lat.toFixed(3)}, ${lng.toFixed(3)}` : "Not selected" },
    { title: "Hub height", data: hubHeight ? `${hubHeight} meters` : "Not selected" },
    { title: "Power curve", data: powerCurve ? `${POWER_CURVE_LABEL[powerCurve] || powerCurve}` : "Not selected" },
  ];

  return (
    <Box sx={{ bgcolor: "background.paper", p: 2, "> *": { color: "text.primary" } }}>
      <Divider textAlign="center" sx={{ my: 2, fontWeight: 600, color: "text.secondary" }}>Summary Results Based on</Divider>
      <Grid container direction="row" spacing={1} marginBottom={2} sx={{ justifyContent: "space-between", alignItems: "stretch" }}>
        {settingOptions.map((option, index) => (
          <Item key={`setting_option_${index}`} sx={{ flexGrow: 1 }}>
            <Typography variant="h6" sx={{ fontSize: "1rem" }}>{option.title}</Typography>
            <Typography variant="body2" sx={{ fontSize: "1rem" }}>{option.data}</Typography>
          </Item>
        ))}
      </Grid>
      <Box sx={{ display: "flex", gap: 1, justifyContent: "flex-end", mb: 2 }}>
        <Button variant="outlined" size="small" startIcon={<SettingsIcon />} onClick={toggleSettings} sx={{ fontSize: "0.9em", textTransform: "none", borderRadius: 2, px: 2, py: 0.5, borderColor: "primary.main", color: "primary.main", "&:hover": { backgroundColor: "primary.main", color: "white", borderColor: "primary.main" } }}>Edit settings</Button>
      </Box>
      <AnalysisResults wind={wind} production={production} />
    </Box>
  );
}
