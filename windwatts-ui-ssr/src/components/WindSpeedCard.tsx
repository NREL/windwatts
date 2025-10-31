"use client";

import { Card, CardContent, Chip, Stack, Typography } from "@mui/material";

function classifyResource(ms: number) {
  if (!Number.isFinite(ms)) return { label: "—", color: "default" as const };
  if (ms >= 7) return { label: "High", color: "success" as const };
  if (ms >= 5) return { label: "Moderate", color: "warning" as const };
  return { label: "Low", color: "warning" as const };
}

export default function WindSpeedCard({ wind }: { wind: any }) {
  const ms = Number(wind?.global_avg ?? 0);
  const cls = classifyResource(ms);
  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={1} direction="row" alignItems="center">
          <Typography variant="h6">Average Wind Speed</Typography>
          <Chip label={cls.label} color={cls.color} size="small" />
        </Stack>
        <Typography variant="h4" sx={{ mt: 1 }}>{Number.isFinite(ms) ? `${ms.toFixed(1)} m/s` : "—"}</Typography>
        <Typography variant="caption" color="text.secondary">Average wind speed at selected height</Typography>
      </CardContent>
    </Card>
  );
}


