"use client";

import { Link, Typography } from "@mui/material";
import { useContext } from "react";
import { SettingsContext } from "../../providers/SettingsContext";
import { DATA_MODEL_INFO } from "../../constants/dataModelInfo";

export const DataSourceLinks = () => {
  const { preferredModel } = useContext(SettingsContext);
  const info = DATA_MODEL_INFO[preferredModel] || DATA_MODEL_INFO.era5;
  return (
    <Typography variant="body2" color="text.secondary">
      * Estimates based on {" "}
      <Link href={info.source_href} target="_blank" rel="noopener noreferrer" underline="hover">
        {info.label} reanalysis data
      </Link>{" "}
      &mdash; {info.help_href && (
        <Link href={info.help_href} target="_blank" rel="noopener noreferrer" underline="hover">
          Learn more
        </Link>
      )}.
    </Typography>
  );
};
