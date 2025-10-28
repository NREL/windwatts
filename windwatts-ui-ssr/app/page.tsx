import { Stack, Typography } from "@mui/material";
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

  const wind = await era5.windspeed({
    lat,
    lng,
    height,
    source: "athena_era5"
  });

  const production = await era5.energyProduction({
    lat,
    lng,
    height,
    powerCurve,
    source: "athena_era5",
    time_period: "summary"
  });

  return (
    <main style={{ padding: 24 }}>
      <Stack spacing={2}>
        <Typography variant="h5">Wind Data</Typography>
        <Typography>
          lat={lat.toFixed(4)}, lng={lng.toFixed(4)}, height={height}m,
          powerCurve={powerCurve}
        </Typography>
        <Typography variant="subtitle1">ERA5 Windspeed</Typography>
        <pre style={{ background: "#f5f5f5", padding: 12, borderRadius: 6 }}>
{JSON.stringify(wind, null, 2)}
        </pre>
        <Typography variant="subtitle1">Energy Production (summary)</Typography>
        <pre style={{ background: "#f5f5f5", padding: 12, borderRadius: 6 }}>
{JSON.stringify(production, null, 2)}
        </pre>
      </Stack>
    </main>
  );
}
