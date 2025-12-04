import { Outlet, Link as RouterLink } from "react-router-dom";
import { Link, Box, AppBar, Toolbar } from "@mui/material";
import { Settings } from "../settings";
import { useContext } from "react";
import { SettingsContext } from "../../providers/SettingsContext";
import { MobileBottomSheet } from "./";
import {
  MobileBottomSheetProvider,
  useMobileBottomSheet,
} from "../../providers/MobileBottomSheetProvider";
import { useGoogleMaps } from "../../hooks";

const LayoutMobileContent = () => {
  const APP_TITLE = import.meta.env.VITE_APP_TITLE || "Wind Watts";
  const { setCurrentPosition } = useContext(SettingsContext);
  const { setBottomSheetRef } = useMobileBottomSheet();

  // Load Google Maps API for SearchBar
  const { isLoaded } = useGoogleMaps();

  const handlePlaceSelected = (place: google.maps.places.PlaceResult) => {
    const lat = place.geometry?.location?.lat();
    const lng = place.geometry?.location?.lng();
    if (lat === undefined || lng === undefined) {
      console.error("Selected place does not have valid coordinates.");
      return;
    }
    setCurrentPosition({ lat, lng });
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <Box sx={{ flexGrow: 0 }}>
        <AppBar position="static" sx={{ bgcolor: "#0279c2" }}>
          <Toolbar sx={{ display: "flex", justifyContent: "space-between" }}>
            <Link
              to="/"
              variant="h5"
              component={RouterLink}
              underline="none"
              sx={{ flexGrow: 1, color: "white" }}
            >
              {APP_TITLE}
            </Link>
            <Box
              component="img"
              sx={{ height: 40 }}
              src="/assets/NLR-Blue-TypeTreatment-Block-Top-RGB.svg"
              alt="NLR Logo"
            />
          </Toolbar>
        </AppBar>
      </Box>

      <Box
        sx={{
          display: "flex",
          flexGrow: 1,
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            position: "relative",
            flexGrow: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <Outlet />
        </Box>

        {/* Show mobile bottom sheet on mobile */}
        <MobileBottomSheet
          ref={setBottomSheetRef}
          isLoaded={isLoaded}
          onPlaceSelected={handlePlaceSelected}
        />
      </Box>
      <Settings />
    </Box>
  );
};

export const LayoutMobile = () => {
  return (
    <MobileBottomSheetProvider>
      <LayoutMobileContent />
    </MobileBottomSheetProvider>
  );
};
