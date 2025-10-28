import { Stack, Typography } from "@mui/material";

export const dynamic = "force-dynamic";

const DENVER = { lat: 39.7392, lng: -104.9903 };

async function getEra5Wind(lat: number, lng: number) {
  const base = process.env.WINDWATTS_API_BASE || (process.env.DOCKER === "1" ? "http://nginx/api" : "http://localhost:8080/api");
  const url = new URL(`${base}/era5/windspeed`);
  url.searchParams.set("lat", String(lat));
  url.searchParams.set("lng", String(lng));
  url.searchParams.set("height", String(40));
  url.searchParams.set("source", "athena_era5");
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ERA5 windspeed: ${res.status}`);
  return res.json();
}

export default async function DenverPage() {
  const data = await getEra5Wind(DENVER.lat, DENVER.lng);
  return (
    <main style={{ padding: 24 }}>
      <Stack spacing={2}>
        <Typography variant="h5">ERA5 Windspeed @40m — Denver, CO</Typography>
        <pre style={{ background: "#f5f5f5", padding: 12, borderRadius: 6 }}>
          {JSON.stringify(data, null, 2)}
        </pre>
      </Stack>
    </main>
  );
}
