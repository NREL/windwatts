"use client";

import { AppBar, Box, Toolbar, Link as MuiLink } from "@mui/material";
import Link from "next/link";

export default function Header() {
  const APP_TITLE = process.env.NEXT_PUBLIC_APP_TITLE || "Wind Watts";
  return (
    <AppBar position="sticky" elevation={0} sx={{ bgcolor: "#0279c2", boxShadow: "none" }}>
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
          src="/NREL-logo-reversed.png"
          alt="NREL Logo"
          sx={{ height: 40, width: "auto", display: "block" }}
        />
      </Toolbar>
    </AppBar>
  );
}


