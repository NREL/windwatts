"use client";

import { useState, memo } from "react";
import { Paper, Typography, Box, Tooltip, IconButton, Button, Collapse } from "@mui/material";
import { InfoOutlined } from "@mui/icons-material";
import { getWindResource } from "../../utils/index"; // re-export or implement as needed

export const WindResourceCard = memo(function WindResourceCard({ windData, loading, error, hasData, outOfBounds, outOfBoundsMessage }: { windData: any; loading?: boolean; error?: boolean; hasData?: boolean; outOfBounds?: boolean; outOfBoundsMessage?: string }) {
  const [expanded, setExpanded] = useState(false);

  const title = "Wind Resource";
  const subheader = "Broad measure of how much wind is available";
  const details: string[] = [];

  if (outOfBounds) {
    return (
      <Paper sx={{ p: 2, minHeight: 100, display: "flex", justifyContent: "center", bgcolor: "warning.light" }}>
        <Typography variant="body2">{outOfBoundsMessage || "Out of bounds"}</Typography>
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 2, minHeight: 100, display: "flex", flexDirection: "column", justifyContent: "center", bgcolor: "error.light", color: "error.contrastText" }}>
        <Typography variant="subtitle2" gutterBottom>{title}</Typography>
        <Typography variant="body2">Error loading data</Typography>
      </Paper>
    );
  }

  if (loading) {
    return (
      <Paper sx={{ p: 2, minHeight: 100, display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>{title}</Typography>
        <Typography variant="body2">Loading…</Typography>
      </Paper>
    );
  }

  if (!hasData) {
    return (
      <Paper sx={{ p: 2, minHeight: 100, display: "flex", flexDirection: "column", justifyContent: "center", bgcolor: "grey.100" }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>{title}</Typography>
        <Typography variant="body2" color="text.secondary">No data available</Typography>
      </Paper>
    );
  }

  const windResourceData = getWindResource(Number(windData?.global_avg ?? 0));
  const getWindResourceInfo = (resource: string) => {
    const r = resource.toLowerCase();
    if (r.includes("high")) return { bgColor: "success.light", textColor: "success.contrastText" };
    if (r.includes("moderate")) return { bgColor: "info.light", textColor: "info.contrastText" };
    return { bgColor: "warning.light", textColor: "warning.contrastText" };
  };
  const windInfo = getWindResourceInfo(windResourceData);

  const handleExpandClick = () => setExpanded((v) => !v);

  return (
    <Paper sx={{ p: 2, minHeight: 100, display: "flex", flexDirection: "column", justifyContent: "center", bgcolor: windInfo.bgColor, color: windInfo.textColor }}>
      <Typography variant="subtitle2" sx={{ opacity: 0.9 }} gutterBottom>{title}</Typography>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography variant="h5" sx={{ fontWeight: "bold" }}>{windResourceData}</Typography>
        <Tooltip title="Wind resource classification" arrow>
          <IconButton size="small" sx={{ color: windInfo.textColor, opacity: 0.8, "&:hover": { opacity: 1, bgcolor: "rgba(255,255,255,0.1)" } }}>
            <InfoOutlined fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="caption" sx={{ opacity: 0.8 }}>{subheader}</Typography>
      {details.length > 0 && (
        <>
          <Button onClick={handleExpandClick} aria-expanded={expanded} aria-label="show details" size="small" sx={{ mt: 1, alignSelf: "flex-start", color: windInfo.textColor, "&:hover": { bgcolor: "rgba(255,255,255,0.1)" } }}>
            {expanded ? "Hide Details" : "Show Details"}
          </Button>
          <Collapse in={expanded} timeout="auto" unmountOnExit>
            <Box sx={{ mt: 2 }}>
              {details.map((detail, i) => (
                <Typography key={`detail_${i}`} variant="body2" sx={{ opacity: 0.9 }} mb={1}>{detail}</Typography>
              ))}
            </Box>
          </Collapse>
        </>
      )}
    </Paper>
  );
});
