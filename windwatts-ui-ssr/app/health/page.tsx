import { Stack, Typography } from "@mui/material";
import { fetchJson } from "../../src/server/api";

export const dynamic = "force-dynamic";

async function getHealth() {
  return fetchJson<{ status: string }>("/healthcheck");
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
