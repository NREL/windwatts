"use client";

import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from "@mui/material";

export default function ProductionDetailsTable({ data }: { data: Record<string, any> }) {
  if (!data) return null;
  const rows = Object.entries(data);
  return (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Period</TableCell>
            <TableCell align="right">Avg wind (m/s)</TableCell>
            <TableCell align="right">kWh produced</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map(([period, vals]) => (
            <TableRow key={period}>
              <TableCell>{period}</TableCell>
              <TableCell align="right">{vals["Average wind speed (m/s)"] ?? vals["Average wind speed, m/s"] ?? "—"}</TableCell>
              <TableCell align="right">{vals["kWh produced"] ?? "—"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}


