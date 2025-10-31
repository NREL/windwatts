"use client";

import { styled } from "@mui/material/styles";
import { ToggleButtonGroup } from "@mui/material";

export const SettingToggleButtonGroup = styled(ToggleButtonGroup)(() => ({
  backgroundColor: "#efefef",
  borderRadius: "5rem",
  "& .MuiToggleButton-root": {
    border: 0,
    padding: "0.125rem 0.75rem",
    textTransform: "none",
    "&:hover": { backgroundColor: "transparent" }
  },
  "& .MuiToggleButton-root.Mui-selected": {
    backgroundColor: "#0279c2",
    fontWeight: "bold",
    borderRadius: "5rem",
    "& p": { color: "white", fontWeight: 500 },
    "&:hover": { backgroundColor: "#0279c2" }
  }
}));
