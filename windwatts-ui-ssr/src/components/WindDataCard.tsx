"use client";

import { Card, CardContent, Stack, Typography, LinearProgress } from "@mui/material";
import { useContext } from "react";
import { UnitsContext } from "../providers/UnitsContext";
import ProductionCards from "./ProductionCards";
import WindSpeedCard from "./WindSpeedCard";
import ProductionDataTable from "./ProductionDataTable";

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

  const monthly = production?.monthly_avg_energy_production;
  const yearly = production?.yearly_avg_energy_production;

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={2}>
          <Typography variant="h6">Results</Typography>
          {!wind || !production ? <LinearProgress /> : null}
          <WindSpeedCard wind={wind} />
          <Typography>Estimated annual energy: {energyDisplay}</Typography>
          {production ? <ProductionCards production={production} /> : null}
          {monthly ? (
            <ProductionDataTable title="Monthly Production Details" data={monthly} timeUnit="month" />
          ) : yearly ? (
            <ProductionDataTable title="Yearly Production Details" data={yearly} timeUnit="year" />
          ) : null}
        </Stack>
      </CardContent>
    </Card>
  );
}
