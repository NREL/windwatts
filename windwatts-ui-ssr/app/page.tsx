import { Box, Stack } from "@mui/material";
import RightPane from "../src/components/RightPane";
import SearchBar from "../src/components/SearchBar";
import GoogleMapsLoader from "../src/components/GoogleMapsLoader";
import Map from "../src/components/Map";
import SettingsModal from "../src/components/SettingsModal";
import { era5 } from "../src/server/api";
import { URL_PARAM_DEFAULTS } from "../src/utils/urlParams";

export const dynamic = "force-dynamic";

export default async function Page({
  searchParams
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const lat = parseFloat((searchParams.lat as string) || "39.7392");
  const lng = parseFloat((searchParams.lng as string) || "-104.9903");
  const height = parseInt(
    (searchParams.hubHeight as string) || String(URL_PARAM_DEFAULTS.hubHeight),
    10
  );
  const powerCurve =
    (searchParams.powerCurve as string) || URL_PARAM_DEFAULTS.powerCurve;

  let wind: any = null;
  let production: any = null;
  let error: string | null = null;
  let curves: string[] = [];

  try {
    const [w, p, c] = await Promise.all([
      era5.windspeed({ lat, lng, height, source: "athena_era5" }),
      era5.energyProduction({
        lat,
        lng,
        height,
        powerCurve,
        source: "athena_era5",
        time_period: "all"
      }),
      era5.availablePowerCurves()
    ]);
    wind = w;
    production = p;
    curves = c?.available_power_curves || [];
  } catch (e: any) {
    error = e?.message || "Failed to load";
  }

  return (
    <main style={{ height: "calc(100vh - 64px)" }}>
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 420px" }, height: "100%" }}>
        {/* Left: Map column with overlay search */}
        <Box sx={{ position: { md: "sticky" }, top: 0, height: { xs: 360, md: "100%" }, borderRight: { md: 1 }, borderColor: "divider" }}>
          <GoogleMapsLoader>
            <Box sx={{ position: "relative", height: "100%", width: "100%" }}>
              <Box sx={{ position: "absolute", top: 16, left: 0, right: 0, display: "flex", justifyContent: "center", px: 2, zIndex: 2, pointerEvents: "none" }}>
                <Box sx={{ width: "100%", maxWidth: 500, bgcolor: "background.paper", borderRadius: 2, p: 1.5, boxShadow: 3, pointerEvents: "auto" }}>
                  <SearchBar />
                </Box>
              </Box>
              <Map height={"100%"} />
            </Box>
          </GoogleMapsLoader>
        </Box>

        {/* Right: Results pane (scrollable) */}
        <Box sx={{ overflow: "auto", p: 2 }}>
          <RightPane powerCurves={curves} wind={wind} production={production} error={error} />
        </Box>
      </Box>
      <SettingsModal powerCurves={curves} />
    </main>
  );
}
