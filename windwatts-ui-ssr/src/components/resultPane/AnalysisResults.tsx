"use client";

import { Box, Stack, Divider, Typography } from "@mui/material";
import { ProductionCard } from "./ProductionCard";
import { WindSpeedCard } from "./WindSpeedCard";
import { WindResourceCard } from "./WindResourceCard";
import { DataSourceLinks } from "./DataSourceLinks";
import { useContext } from "react";
import { SettingsContext } from "../../providers/SettingsContext";

export default function AnalysisResults({ wind, production }: { wind: any; production: any }) {
  const { ensemble, preferredModel } = useContext(SettingsContext);

  return (
    <Stack spacing={2}>
      {ensemble ? (
        <>
          <Divider textAlign="center" sx={{ my: 1, fontWeight: 600, color: "text.secondary" }}>Ensemble Model Results *</Divider>
          <Typography variant="body2" color="text.secondary">* Experimental model (under development)</Typography>
        </>
      ) : (
        <>
          <Box sx={{ display: "flex", gap: 2 }}>
            <Box sx={{ flex: 1 }}>
              <WindSpeedCard windData={wind} loading={!wind} hasData={!!wind} />
            </Box>
            <Box sx={{ flex: 1 }}>
              <WindResourceCard windData={wind} loading={!wind} hasData={!!wind} />
            </Box>
          </Box>
          <DataSourceLinks />
        </>
      )}
      <ProductionCard productionData={production} />
    </Stack>
  );
}
