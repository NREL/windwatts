"use client";

import { Box, MenuItem, Stack, TextField } from "@mui/material";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

export default function Controls({ powerCurves }: { powerCurves: string[] }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const spHeight = Number(searchParams.get("hubHeight") || 40);
  const [heightInput, setHeightInput] = useState<string>(String(spHeight));

  const powerCurve = searchParams.get("powerCurve") || (powerCurves[0] || "nrel-reference-2.5kW");

  const updateParam = (key: string, value: string) => {
    const next = new URLSearchParams(searchParams as any);
    next.set(key, value);
    router.replace(`${pathname}?${next.toString()}`);
  };

  const commitHeight = () => {
    const num = Math.max(10, Math.min(200, Number(heightInput)));
    updateParam("hubHeight", String(Math.round(num)));
  };

  return (
    <Box>
      <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
        <TextField
          label="Hub Height (m)"
          type="number"
          value={heightInput}
          inputProps={{ min: 10, max: 200 }}
          onChange={(e) => setHeightInput(e.target.value)}
          onBlur={commitHeight}
        />
        <TextField
          select
          label="Power Curve"
          value={powerCurve}
          onChange={(e) => updateParam("powerCurve", e.target.value)}
        >
          {powerCurves.map((pc) => (
            <MenuItem key={pc} value={pc}>
              {pc}
            </MenuItem>
          ))}
        </TextField>
      </Stack>
    </Box>
  );
}
