"use client";

import { useContext } from "react";
import { Dialog } from "@mui/material";
import { SettingsContext } from "../providers/SettingsContext";
import { Settings as SettingsContent } from "./settings/Settings";

export default function SettingsModal({ powerCurves = [] }: { powerCurves?: string[] }) {
  const { settingsOpen, toggleSettings } = useContext(SettingsContext);
  return (
    <Dialog open={!!settingsOpen} onClose={toggleSettings} fullWidth maxWidth="sm">
      <SettingsContent powerCurves={powerCurves} />
    </Dialog>
  );
}
