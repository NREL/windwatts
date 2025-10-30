"use client";

import { AppBar, Box, Toolbar, Link as MuiLink } from "@mui/material";
import Link from "next/link";

export default function Header() {
  const APP_TITLE = process.env.NEXT_PUBLIC_APP_TITLE || "Wind Watts";
  return (
    <AppBar position="sticky" sx={{ bgcolor: "#0279c2" }}>
      <Toolbar sx={{ display: "flex", justifyContent: "space-between" }}>
        <MuiLink
          href="/"
          component={Link}
          underline="none"
          variant="h5"
          sx={{ flexGrow: 1, color: "white" }}
        >
          {APP_TITLE}
        </MuiLink>
        <Box
          component="img"
          sx={{ height: 40 }}
          src="/NREL-logo-reversed.png"
          alt="NREL Logo"
          onError={(e: any) => {
            // hide if not present yet
            e.currentTarget.style.display = "none";
          }}
        />
      </Toolbar>
    </AppBar>
  );
}


