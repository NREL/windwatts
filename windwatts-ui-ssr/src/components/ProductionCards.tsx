"use client";

import { Grid, Paper, Stack, Typography } from "@mui/material";
import { useContext } from "react";
import { UnitsContext } from "../providers/UnitsContext";

function formatNumber(num: number, digits = 1) {
  return Number(num).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

function toDisplay(valueKwh: number, unit: string) {
  if (unit === "MWh") return `${formatNumber(valueKwh / 1000, 1)} MWh`;
  return `${formatNumber(valueKwh, 0)} kWh`;
}

export default function ProductionCards({ production }: { production: any }) {
  const { units } = useContext(UnitsContext);
  const summary = production?.summary_avg_energy_production || {};
  const avg = Number(summary?.["Average year"]?.["kWh produced"] || 0);
  const high = Number(summary?.["Highest year"]?.["kWh produced"] || 0);
  const low = Number(summary?.["Lowest year"]?.["kWh produced"] || 0);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sm={4}>
        <Paper sx={{ p: 2, textAlign: "center", bgcolor: "primary.light", color: "primary.contrastText" }}>
          <Typography variant="caption" sx={{ opacity: 0.9 }}>Average</Typography>
          <Typography variant="h5" sx={{ fontWeight: "bold", mt: 0.5 }}>
            {toDisplay(avg, units.output)}
          </Typography>
        </Paper>
      </Grid>
      <Grid item xs={12} sm={4}>
        <Paper sx={{ p: 2, textAlign: "center", bgcolor: "success.light", color: "success.contrastText" }}>
          <Typography variant="caption" sx={{ opacity: 0.9 }}>Highest Year</Typography>
          <Typography variant="h6" sx={{ fontWeight: "bold", mt: 0.5 }}>
            {toDisplay(high, units.output)}
          </Typography>
        </Paper>
      </Grid>
      <Grid item xs={12} sm={4}>
        <Paper sx={{ p: 2, textAlign: "center", bgcolor: "warning.light", color: "warning.contrastText" }}>
          <Typography variant="caption" sx={{ opacity: 0.9 }}>Lowest Year</Typography>
          <Typography variant="h6" sx={{ fontWeight: "bold", mt: 0.5 }}>
            {toDisplay(low, units.output)}
          </Typography>
        </Paper>
      </Grid>
    </Grid>
  );
}
