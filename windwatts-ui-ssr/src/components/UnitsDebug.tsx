"use client";

import { useContext } from "react";
import { Typography } from "@mui/material";
import { UnitsContext } from "../providers/UnitsContext";

export default function UnitsDebug() {
  const { units } = useContext(UnitsContext);
  return (
    <Typography>
      Units: windspeed={units.windspeed}, output={units.output}
    </Typography>
  );
}
