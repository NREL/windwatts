import { useContext, useState, memo } from "react";
import {
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Button,
  Collapse,
  Typography,
  Box,
  Divider,
  Chip,
  Paper,
  Skeleton,
  Link,
} from "@mui/material";
import { UnitsContext } from "../../providers/UnitsContext";
import {
  KEY_AVERAGE_YEAR,
  KEY_HIGHEST_YEAR,
  KEY_KWH_PRODUCED,
  KEY_LOWEST_YEAR,
  DATA_MODEL_INFO,
} from "../../constants";
import { ProductionDataTable } from "./ProductionDataTable";
import { convertOutput, getOutOfBoundsMessage } from "../../utils";
import { ExpandMore, ExpandLess } from "@mui/icons-material";
import { OutOfBoundsWarning } from "../shared";
import { useProductionData } from "../../hooks";
// import { SettingsContext } from "../../providers/SettingsContext";

export const ProductionCard = memo(() => {
  const [expanded, setExpanded] = useState(false);
  const { units } = useContext(UnitsContext);
  // const { preferredModel } = useContext(SettingsContext);
  const {
    productionData,
    isLoading,
    error,
    hasData,
    outOfBounds,
    dataModel,
    lat,
    lng,
  } = useProductionData();

  const title = "Production";
  // Always show ERA5 in the subheader since production always uses ERA5 data
  const productionModelDisplay = dataModel.split('-')[0].toUpperCase();
  const subheader = (
    <>
      Estimated Annual Production Potential from{" "}
      <Link
        href={
          DATA_MODEL_INFO[dataModel]?.source_href ||
          DATA_MODEL_INFO['era5-quantiles'].source_href
        }
        target="_blank"
        rel="noopener noreferrer"
        underline="hover"
      >
        {productionModelDisplay} Model
      </Link>
    </>
  );
  const details = [
    "Wind energy production can vary significantly from year to year. Understanding both the average resource and its variability is key to setting realistic expectations.",
  ];

  const handleExpandClick = () => {
    setExpanded(!expanded);
  };

  // Out-of-bounds state
  if (outOfBounds) {
    return (
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              Production
              <Chip
                label="Primary Analysis"
                size="small"
                color="warning"
                variant="outlined"
              />
            </Box>
          }
          subheader={subheader}
          sx={{ bgcolor: "warning.light", pb: 1 }}
        />
        <CardContent sx={{ py: 2 }}>
          <OutOfBoundsWarning
            message={getOutOfBoundsMessage(lat, lng, dataModel)}
          />
        </CardContent>
      </Card>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              {title}
              <Chip
                label="Primary Analysis"
                size="small"
                color="primary"
                variant="outlined"
              />
            </Box>
          }
          subheader={subheader}
          sx={{ bgcolor: "var(--color-light)", pb: 1 }}
        />

        <CardContent sx={{ pb: 2 }}>
          {/* Skeleton for production metrics */}
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "1fr",
                sm: "1fr",
                md: "repeat(3, 1fr)",
              },
              gap: 2,
              mb: 3,
            }}
          >
            {[1, 2, 3].map((index) => (
              <Paper key={index} sx={{ p: 2, textAlign: "center" }}>
                <Skeleton
                  variant="text"
                  width="60%"
                  height={20}
                  sx={{ mx: "auto" }}
                />
                <Skeleton
                  variant="text"
                  width="80%"
                  height={40}
                  sx={{ mx: "auto", mt: 0.5 }}
                />
                <Skeleton
                  variant="text"
                  width="40%"
                  height={16}
                  sx={{ mx: "auto" }}
                />
              </Paper>
            ))}
          </Box>

          {/* Skeleton for details */}
          <Skeleton variant="text" width="100%" height={20} />
          <Skeleton variant="text" width="90%" height={20} />
        </CardContent>

        <Divider sx={{ borderStyle: "dotted" }} />

        <CardActions sx={{ justifyContent: "space-between", px: 2 }}>
          <Skeleton variant="text" width="150px" height={20} />
          <Skeleton variant="rectangular" width="60px" height={32} />
        </CardActions>
      </Card>
    );
  }

  // Error state
  if (error) {
    return (
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              {title}
              <Chip
                label="Primary Analysis"
                size="small"
                color="error"
                variant="outlined"
              />
            </Box>
          }
          subheader={subheader}
          sx={{ bgcolor: "var(--color-light)", pb: 1 }}
        />

        <CardContent sx={{ py: 4, textAlign: "center" }}>
          <Typography color="error" variant="h6" gutterBottom>
            Error Loading Production Data
          </Typography>
          <Typography color="text.secondary" variant="body2">
            Unable to load production analysis. Please check your settings and
            try again.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (!hasData) {
    return (
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              {title}
              <Chip
                label="Primary Analysis"
                size="small"
                color="default"
                variant="outlined"
              />
            </Box>
          }
          subheader={subheader}
          sx={{ bgcolor: "var(--color-light)", pb: 1 }}
        />

        <CardContent sx={{ py: 4, textAlign: "center" }}>
          <Typography color="text.secondary" variant="h6" gutterBottom>
            No Production Data Available
          </Typography>
          <Typography color="text.secondary" variant="body2">
            Please set your location and turbine settings to see production
            analysis.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Data loaded successfully
  const summaryData = productionData?.summary_avg_energy_production;
  const avgProduction = Number(
    summaryData?.[KEY_AVERAGE_YEAR]?.[KEY_KWH_PRODUCED] || 0
  );
  const lowProduction = Number(
    summaryData?.[KEY_LOWEST_YEAR]?.[KEY_KWH_PRODUCED] || 0
  );
  const highProduction = Number(
    summaryData?.[KEY_HIGHEST_YEAR]?.[KEY_KWH_PRODUCED] || 0
  );

  // Determine data type and set up variables
  // may need something more robust here if we add more data types
  const monthlyData = "monthly_avg_energy_production" in productionData;
  const tableData = monthlyData
    ? productionData.monthly_avg_energy_production
    : productionData?.yearly_avg_energy_production;

  const detailsLabel = monthlyData
    ? "Monthly Production Details"
    : "Yearly Production Details";

  const tableTitle = "";

  return (
    <Card>
      <CardHeader
        title={
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            {title}
            <Chip
              label="Primary Analysis"
              size="small"
              color="primary"
              variant="outlined"
            />
          </Box>
        }
        subheader={subheader}
        sx={{ bgcolor: "var(--color-light)", pb: 1 }}
      />

      {/* Key Production Metrics - Always Visible */}
      <CardContent sx={{ pb: 2 }}>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "1fr",
              sm: "1fr",
              md: "repeat(3, 1fr)",
            },
            gap: 2,
            mb: 3,
          }}
        >
          <Paper
            sx={{
              p: 2,
              textAlign: "center",
              bgcolor: "primary.light",
              color: "primary.contrastText",
            }}
          >
            <Typography variant="caption" sx={{ opacity: 0.9 }}>
              Average
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: "bold", mt: 0.5 }}>
              {convertOutput(Number(avgProduction), units.output).replace(
                /\s\w+$/,
                ""
              )}
            </Typography>
            <Typography variant="caption">{units.output}</Typography>
          </Paper>

          <Paper
            sx={{
              p: 2,
              textAlign: "center",
              bgcolor: "success.light",
              color: "success.contrastText",
            }}
          >
            <Typography variant="caption" sx={{ opacity: 0.9 }}>
              Highest Year
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: "bold", mt: 0.5 }}>
              {convertOutput(Number(highProduction), units.output).replace(
                /\s\w+$/,
                ""
              )}
            </Typography>
            <Typography variant="caption">{units.output}</Typography>
          </Paper>

          <Paper
            sx={{
              p: 2,
              textAlign: "center",
              bgcolor: "warning.light",
              color: "warning.contrastText",
            }}
          >
            <Typography variant="caption" sx={{ opacity: 0.9 }}>
              Lowest Year
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: "bold", mt: 0.5 }}>
              {convertOutput(Number(lowProduction), units.output).replace(
                /\s\w+$/,
                ""
              )}
            </Typography>
            <Typography variant="caption">{units.output}</Typography>
          </Paper>
        </Box>

        {details.map((detail, index) => (
          <Typography
            mb={1}
            key={title + "result_detail" + index}
            variant="body2"
            color="text.secondary"
          >
            {detail}
          </Typography>
        ))}
      </CardContent>

      <Divider sx={{ borderStyle: "dotted" }} />

      {/* Detailed Breakdown - Expandable */}
      <CardActions
        sx={{
          justifyContent: { xs: "flex-end", sm: "space-between" },
          px: 2,
        }}
      >
        <Typography
          variant="body1"
          color="text.secondary"
          sx={{
            display: { xs: "none", sm: "block" }, // Hide on mobile to save space
          }}
        >
          {detailsLabel}
        </Typography>
        <Button
          onClick={handleExpandClick}
          aria-expanded={expanded}
          aria-label="show detailed breakdown"
          variant="outlined"
          size="small"
          startIcon={expanded ? <ExpandLess /> : <ExpandMore />}
          sx={{
            whiteSpace: "nowrap",
            minWidth: "auto",
            px: 2,
          }}
        >
          {expanded ? "Hide" : "Show"}
        </Button>
      </CardActions>

      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <CardContent sx={{ pt: 0 }}>
          {tableData && (
            <ProductionDataTable
              title={tableTitle}
              data={tableData}
              timeUnit={monthlyData ? "month" : "year"}
            />
          )}
        </CardContent>
      </Collapse>
    </Card>
  );
});
