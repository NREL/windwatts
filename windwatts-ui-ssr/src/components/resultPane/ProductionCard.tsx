"use client";

import { useContext, useState, memo } from "react";
import { Card, CardHeader, CardContent, CardActions, Button, Collapse, Typography, Box, Divider, Chip, Paper, Link } from "@mui/material";
import { UnitsContext } from "../../providers/UnitsContext";
import { KEY_AVERAGE_YEAR, KEY_HIGHEST_YEAR, KEY_KWH_PRODUCED, KEY_LOWEST_YEAR } from "../../constants/dataKeys";
import ProductionDataTable from "../ProductionDataTable";
import { convertOutput } from "../../utils/units";
import { ExpandMore, ExpandLess } from "@mui/icons-material";
import { SettingsContext } from "../../providers/SettingsContext";
import { DATA_MODEL_INFO } from "../../constants/dataModelInfo";

export const ProductionCard = memo(function ProductionCard({ productionData }: { productionData: any }) {
  const [expanded, setExpanded] = useState(false);
  const { units } = useContext(UnitsContext);
  const { preferredModel } = useContext(SettingsContext);

  const title = "Production";
  const subheader = (
    <>
      Estimated Annual Production Potential from {" "}
      <Link href={DATA_MODEL_INFO[preferredModel]?.source_href || DATA_MODEL_INFO.era5.source_href} target="_blank" rel="noopener noreferrer" underline="hover">{preferredModel.toUpperCase()} Model</Link>
    </>
  );
  const details = ["Wind energy production can vary significantly from year to year. Understanding both the average resource and its variability is key to setting realistic expectations."];

  if (!productionData) return null;

  const summaryData = productionData?.summary_avg_energy_production;
  const avgProduction = Number(summaryData?.[KEY_AVERAGE_YEAR]?.[KEY_KWH_PRODUCED] || 0);
  const lowProduction = Number(summaryData?.[KEY_LOWEST_YEAR]?.[KEY_KWH_PRODUCED] || 0);
  const highProduction = Number(summaryData?.[KEY_HIGHEST_YEAR]?.[KEY_KWH_PRODUCED] || 0);
  const monthlyData = "monthly_avg_energy_production" in productionData;
  const tableData = monthlyData ? productionData.monthly_avg_energy_production : productionData?.yearly_avg_energy_production;
  const detailsLabel = monthlyData ? "Monthly Production Details" : "Yearly Production Details";

  return (
    <Card>
      <CardHeader
        title={
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            {title}
            <Chip label="Primary Analysis" size="small" color="primary" variant="outlined" />
          </Box>
        }
        subheader={subheader}
        sx={{ bgcolor: "var(--color-light)", pb: 1 }}
      />
      <CardContent sx={{ pb: 2 }}>
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "1fr", md: "repeat(3, 1fr)" }, gap: 2, mb: 3 }}>
          <Paper sx={{ p: 2, textAlign: "center", bgcolor: "primary.light", color: "primary.contrastText" }}>
            <Typography variant="caption" sx={{ opacity: 0.9 }}>Average</Typography>
            <Typography variant="h5" sx={{ fontWeight: "bold", mt: 0.5 }}>{convertOutput(avgProduction, units.output).replace(/\s\w+$/, "")}</Typography>
            <Typography variant="caption">{units.output}</Typography>
          </Paper>
          <Paper sx={{ p: 2, textAlign: "center", bgcolor: "success.light", color: "success.contrastText" }}>
            <Typography variant="caption" sx={{ opacity: 0.9 }}>Highest Year</Typography>
            <Typography variant="h6" sx={{ fontWeight: "bold", mt: 0.5 }}>{convertOutput(highProduction, units.output).replace(/\s\w+$/, "")}</Typography>
            <Typography variant="caption">{units.output}</Typography>
          </Paper>
          <Paper sx={{ p: 2, textAlign: "center", bgcolor: "warning.light", color: "warning.contrastText" }}>
            <Typography variant="caption" sx={{ opacity: 0.9 }}>Lowest Year</Typography>
            <Typography variant="h6" sx={{ fontWeight: "bold", mt: 0.5 }}>{convertOutput(lowProduction, units.output).replace(/\s\w+$/, "")}</Typography>
            <Typography variant="caption">{units.output}</Typography>
          </Paper>
        </Box>
        {details.map((detail, index) => (
          <Typography mb={1} key={`detail_${index}`} variant="body2" color="text.secondary">{detail}</Typography>
        ))}
      </CardContent>
      <Divider sx={{ borderStyle: "dotted" }} />
      <CardActions sx={{ justifyContent: { xs: "flex-end", sm: "space-between" }, px: 2 }}>
        <Typography variant="body1" color="text.secondary" sx={{ display: { xs: "none", sm: "block" } }}>{detailsLabel}</Typography>
        <Button onClick={() => setExpanded((v) => !v)} aria-expanded={expanded} aria-label="show detailed breakdown" variant="outlined" size="small" startIcon={expanded ? <ExpandLess /> : <ExpandMore />} sx={{ whiteSpace: "nowrap", minWidth: "auto", px: 2 }}>
          {expanded ? "Hide" : "Show"}
        </Button>
      </CardActions>
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <CardContent sx={{ pt: 0 }}>
          {tableData && (
            <ProductionDataTable title="" data={tableData} timeUnit={monthlyData ? "month" : "year"} />
          )}
        </CardContent>
      </Collapse>
    </Card>
  );
});
