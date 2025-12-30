"use client";

import { memo, useContext } from "react";
import { Typography } from "@mui/material";
import { UnitsContext } from "../../providers/UnitsContext";
import { convertWindspeed } from "../../utils/units";
import { DataCard, EmptyCard, ErrorCard, LoadingCard, OutOfBoundsCard } from "../shared/CardStates";

export const WindSpeedCard = memo(function WindSpeedCard({ windData, loading, error, hasData, outOfBounds, outOfBoundsMessage }: { windData: any; loading?: boolean; error?: boolean; hasData?: boolean; outOfBounds?: boolean; outOfBoundsMessage?: string }) {
  const { units } = useContext(UnitsContext);
  const title = "Average Wind Speed *";
  const subheader = "Average wind speed at selected height";

  if (outOfBounds) return <OutOfBoundsCard message={outOfBoundsMessage || "Out of bounds"} />;
  if (error) return <ErrorCard title={title} />;
  if (loading) return <LoadingCard title={title} />;
  if (!hasData) return <EmptyCard title={title} />;

  const windSpeedData = convertWindspeed(Number(windData?.global_avg ?? 0), units.windspeed);

  return (
    <DataCard title={title}>
      <Typography variant="h5" color="primary" sx={{ fontWeight: "bold" }}>{windSpeedData}</Typography>
      <Typography variant="caption" color="text.secondary">{subheader}</Typography>
    </DataCard>
  );
});
