"use client";

import ThemeRegistry from "../src/ThemeRegistry";
import { UnitsProvider } from "../src/providers/UnitsProvider";
import { SettingsProvider } from "../src/providers/SettingsProvider";

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeRegistry>
      <UnitsProvider>
        <SettingsProvider>{children}</SettingsProvider>
      </UnitsProvider>
    </ThemeRegistry>
  );
}
