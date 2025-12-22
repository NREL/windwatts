import { memo, useContext } from "react";
import { Box, Paper, Typography, Skeleton } from "@mui/material";
import { UnitsContext } from "../../providers/UnitsContext";
import { SettingsContext } from "../../providers/SettingsContext";
import { useEnsembleTilesData } from "../../hooks";
import { convertWindspeed, convertOutput, getWindResource } from "../../utils";

// Compact, card-less variant for embedding in the top row - using ensemble model
export const EnsembleTiles = memo(() => {
  const { units } = useContext(UnitsContext);
  const { preferredModel } = useContext(SettingsContext);

  const {
    windData,
    productionData,
    isLoading: isTilesLoading,
    error,
    hasData,
  } = useEnsembleTilesData();

  if (preferredModel !== "ensemble") return null;

  const loading = isTilesLoading;
  const hasDataCombined = hasData;
  const windResource = getWindResource(windData?.global_avg ?? 0);

  const getWindResourceInfo = (resource: string) => {
    const resourceLower = resource.toLowerCase();
    if (resourceLower.includes("high")) {
      return {
        bgColor: "success.light",
        textColor: "success.contrastText",
      };
    } else if (resourceLower.includes("moderate")) {
      return {
        bgColor: "info.light",
        textColor: "info.contrastText",
      };
    }
    return {
      bgColor: "warning.light",
      textColor: "warning.contrastText",
    };
  };
  const windInfo = getWindResourceInfo(windResource);

  if (loading) {
    return (
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "repeat(3, 1fr)" },
          gap: 2,
        }}
      >
        {[0, 1, 2].map((i) => (
          <Paper key={i} sx={{ p: 2, textAlign: "center" }} elevation={2}>
            <Skeleton
              variant="text"
              width="40%"
              height={20}
              sx={{ mx: "auto" }}
            />
            <Skeleton
              variant="text"
              width="70%"
              height={36}
              sx={{ mx: "auto", mt: 1 }}
            />
            <Skeleton
              variant="text"
              width="30%"
              height={16}
              sx={{ mx: "auto" }}
            />
          </Paper>
        ))}
      </Box>
    );
  }

  if (error || !hasDataCombined) {
    return null;
  }

  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: { xs: "1fr", sm: "repeat(3, 1fr)" },
        gap: 2,
        alignItems: "stretch",
      }}
    >
      {/* Wind Speed */}
      <Paper sx={{ p: 2, textAlign: "center" }} elevation={2}>
        <Typography variant="subtitle2" color="text.secondary">
          Wind Speed
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {convertWindspeed(windData.global_avg, units.windspeed)}
        </Typography>
      </Paper>

      {/* Wind Resource */}
      <Paper
        sx={{
          p: 2,
          textAlign: "center",
          bgcolor: windInfo.bgColor,
          color: windInfo.textColor,
        }}
        elevation={2}
      >
        <Typography variant="subtitle2" sx={{ opacity: 0.9 }}>
          Wind Resource
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {windResource}
        </Typography>
      </Paper>

      {/* Production */}
      <Paper
        sx={{
          p: 2,
          textAlign: "center",
          bgcolor: "primary.light",
          color: "primary.contrastText",
        }}
        elevation={2}
      >
        <Typography variant="subtitle2" sx={{ opacity: 0.9 }}>
          Production
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {convertOutput(
            Number(productionData?.energy_production || 0),
            units.output
          ).replace(/\s\w+$/, "")}
        </Typography>
        <Typography variant="caption">{units.output}</Typography>
      </Paper>
    </Box>
  );
});
