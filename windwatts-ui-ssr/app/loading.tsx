import { Box, LinearProgress, Stack, Typography } from "@mui/material";

export default function Loading() {
  return (
    <main style={{ padding: 24 }}>
      <Stack spacing={2}>
        <Typography variant="h6">Loading…</Typography>
        <Box>
          <LinearProgress />
        </Box>
      </Stack>
    </main>
  );
}
