"use client";

import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography, Box } from "@mui/material";
import { useContext, useMemo } from "react";
import { UnitsContext } from "../providers/UnitsContext";
import { convertOutput, convertWindspeed } from "../utils/units";
import { KEY_AVG_WIND_SPEED, KEY_KWH_PRODUCED } from "../constants/dataKeys";
import { MONTH_NAMES } from "../constants/ui";

export default function ProductionDataTable({
  title,
  data,
  timeUnit = "month"
}: {
  title: string;
  data: Record<string, Record<string, string | number | null>>;
  timeUnit?: "month" | "year";
}) {
  const { units } = useContext(UnitsContext);

  const rows = useMemo(() => {
    const order = timeUnit === "month" ? MONTH_NAMES : Object.keys(data).sort((a, b) => parseInt(a) - parseInt(b));
    return order.map((time) => ({ time, values: data[time] })).filter((r) => !!r.values);
  }, [data, timeUnit]);

  const windSpeeds = rows.map(({ values }) => Number(values[KEY_AVG_WIND_SPEED]));
  const minWindSpeed = Math.min(...windSpeeds);
  const maxWindSpeed = Math.max(...windSpeeds);
  const windSpeedRange = maxWindSpeed - minWindSpeed;
  const productions = rows.map(({ values }) => Number(values[KEY_KWH_PRODUCED]));
  const maxProduction = Math.max(...productions);

  if (!data || rows.length === 0) {
    return (
      <>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          No data available
        </Typography>
      </>
    );
  }

  return (
    <>
      <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
        {title}
      </Typography>
      <TableContainer component={Box} sx={{ width: "100%", overflowX: { xs: "auto", md: "visible" } }}>
        <Table size="small" sx={{ width: "100%" }}>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: "bold", width: "10%" }}>{timeUnit === "month" ? "Month" : "Year"}</TableCell>
              <TableCell align="right" sx={{ fontWeight: "bold", width: "50%" }}>Wind Speed ({units.windspeed})</TableCell>
              <TableCell align="right" sx={{ fontWeight: "bold", width: "40%" }}>Energy ({units.output})</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map(({ time, values }) => {
              const windSpeed = Number(values[KEY_AVG_WIND_SPEED]);
              const production = Number(values[KEY_KWH_PRODUCED]);
              const windPct = windSpeedRange > 0 ? 30 + ((windSpeed - minWindSpeed) / windSpeedRange) * 70 : 75;
              const energyPct = maxProduction > 0 ? 30 + (production / maxProduction) * 70 : 30;
              return (
                <TableRow key={time} hover>
                  <TableCell sx={{ fontWeight: "medium", whiteSpace: "nowrap" }}>{time}</TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 1, minWidth: 0 }}>
                      <Box sx={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "flex-end", minWidth: 0, maxWidth: { xs: 60, sm: 80, md: 100, lg: 120, xl: 150 } }}>
                        <Box sx={{ width: `${windPct}%`, height: 6, bgcolor: "primary.light", borderRadius: 1, minWidth: 8 }} />
                      </Box>
                      <Box sx={{ minWidth: { xs: 40, sm: 45, md: 50 }, textAlign: "right", fontSize: { xs: "0.75rem", sm: "0.8rem", md: "0.875rem" }, flexShrink: 0, whiteSpace: "nowrap" }}>
                        {convertWindspeed(windSpeed, units.windspeed).replace(/\s\w+\/?\w*$/, "")}
                      </Box>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 1, minWidth: 0 }}>
                      <Box sx={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "flex-end", minWidth: 0, maxWidth: { xs: 60, sm: 80, md: 100, lg: 120, xl: 150 } }}>
                        <Box sx={{ width: `${energyPct}%`, height: 6, bgcolor: "success.light", borderRadius: 1, minWidth: 8 }} />
                      </Box>
                      <Box sx={{ minWidth: { xs: 45, sm: 50, md: 55 }, textAlign: "right", fontSize: { xs: "0.75rem", sm: "0.8rem", md: "0.875rem" }, flexShrink: 0, whiteSpace: "nowrap" }}>
                        {convertOutput(production, units.output).replace(/\s\w+$/, "")}
                      </Box>
                    </Box>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
}
