"use client";

import { PropsWithChildren } from "react";
import { useLoadScript } from "@react-google-maps/api";
import { Box, LinearProgress } from "@mui/material";

const libraries: ("places" | "marker")[] = ["places", "marker"];

export default function GoogleMapsLoader({ children }: PropsWithChildren) {
  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: process.env.NEXT_PUBLIC_MAP_API_KEY || "",
    libraries,
    version: "quarterly"
  });

  if (loadError) return null;
  if (!isLoaded) {
    return (
      <Box sx={{ p: 1 }}>
        <LinearProgress />
      </Box>
    );
  }
  return <>{children}</>;
}
