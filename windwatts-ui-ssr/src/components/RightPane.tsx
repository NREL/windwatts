"use client";

import { Box, Stack, Typography } from "@mui/material";
import SummaryBar from "./SummaryBar";
import Controls from "./Controls";
import WindDataCard from "./WindDataCard";

export default function RightPane({ powerCurves, wind, production, error }: { powerCurves: string[]; wind: any; production: any; error?: string | null }) {
  return (
    <Box sx={{ bgcolor: "background.paper" }}>
      <Stack spacing={2}>
        <Typography variant="h5">Wind Data</Typography>
        <SummaryBar />
        <Controls powerCurves={powerCurves} />
        {error ? (
          <Typography color="error">{error}</Typography>
        ) : (
          <WindDataCard wind={wind} production={production} />
        )}
      </Stack>
    </Box>
  );
}
