import { Stack, Typography } from "@mui/material";

export const dynamic = "force-dynamic";

async function getHealth() {
  const base = process.env.WINDWATTS_API_BASE || (process.env.DOCKER === "1" ? "http://nginx/api" : "http://localhost:8080/api");
  const res = await fetch(`${base}/healthcheck`, {
    cache: "no-store",
    headers: {}
  });
  if (!res.ok) throw new Error("Failed to fetch healthcheck");
  return res.json() as Promise<{ status: string }>;
}

export default async function HealthPage() {
  const data = await getHealth();
  return (
    <main style={{ padding: 24 }}>
      <Stack spacing={2}>
        <Typography variant="h5">API Health</Typography>
        <Typography>Status: {data.status}</Typography>
      </Stack>
    </main>
  );
}
