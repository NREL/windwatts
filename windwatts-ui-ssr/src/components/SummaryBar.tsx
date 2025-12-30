"use client";

import { Card, CardContent, Grid, Stack, Typography } from "@mui/material";
import { useSearchParams } from "next/navigation";

export default function SummaryBar() {
  const sp = useSearchParams();
  const lat = sp.get("lat") || "";
  const lng = sp.get("lng") || "";
  const hub = sp.get("hubHeight") || "40";
  const pc = sp.get("powerCurve") || "nrel-reference-2.5kW";
  return (
    <Card variant="outlined">
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={4}>
            <Stack>
              <Typography variant="caption">Location</Typography>
              <Typography>{lat && lng ? `${lat}, ${lng}` : "—"}</Typography>
            </Stack>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Stack>
              <Typography variant="caption">Hub height</Typography>
              <Typography>{hub} meters</Typography>
            </Stack>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Stack>
              <Typography variant="caption">Power curve</Typography>
              <Typography>{pc}</Typography>
            </Stack>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}


