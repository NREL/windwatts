"use client";

import { Card, CardContent, Typography, Paper } from "@mui/material";

export function LoadingCard({ title }: { title: string }) {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle2" color="text.secondary">{title}</Typography>
      <Typography variant="body2">Loading…</Typography>
    </Paper>
  );
}

export function ErrorCard({ title }: { title: string }) {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle2" color="error">{title}</Typography>
      <Typography variant="body2" color="error">Error loading data</Typography>
    </Paper>
  );
}

export function EmptyCard({ title }: { title: string }) {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle2" color="text.secondary">{title}</Typography>
      <Typography variant="body2" color="text.secondary">No data available</Typography>
    </Paper>
  );
}

export function OutOfBoundsCard({ message }: { message: string }) {
  return (
    <Paper sx={{ p: 2, bgcolor: "warning.light" }}>
      <Typography variant="body2">{message}</Typography>
    </Paper>
  );
}

export function DataCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>{title}</Typography>
        {children}
      </CardContent>
    </Card>
  );
}
