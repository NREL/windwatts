"use client";

import { Card, CardContent, Stack, Typography, LinearProgress } from "@mui/material";
import { useContext } from "react";
import { UnitsContext } from "../providers/UnitsContext";
import ProductionCards from "./ProductionCards";

function formatNumber(num: number, digits = 1) {
  return Number(num).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function WindDataCard({ wind, production }: { wind: any; production: any }) {
  const { units } = useContext(UnitsContext);

  const globalWind = wind?.global_avg ?? wind?.yearly_avg?.["Average year"]?.["Average wind speed (m/s)"] ?? null;
  const energyKwh = production?.energy_production ?? production?.summary_avg_energy_production?.["Average year"]?.["kWh produced"] ?? null;

  const windDisplay = globalWind != null ? `${formatNumber(Number(globalWind), 1)} m/s` : "—";
  const energyDisplay = (() => {
    if (energyKwh == null) return "—";
    if (units.output === "MWh") return `${formatNumber(Number(energyKwh) / 1000, 1)} MWh`;
    return `${formatNumber(Number(energyKwh), 0)} kWh`;
  })();

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={2}>
          <Typography variant="h6">Results</Typography>
          {!wind || !production ? <LinearProgress /> : null}
          <Typography>Average wind speed: {windDisplay}</Typography>
          <Typography>Estimated annual energy: {energyDisplay}</Typography>
          {production ? <ProductionCards production={production} /> : null}
        </Stack>
      </CardContent>
    </Card>
  );
}
