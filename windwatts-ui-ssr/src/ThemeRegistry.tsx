"use client";

import * as React from "react";
import { useServerInsertedHTML } from "next/navigation";
import { CacheProvider } from "@emotion/react";
import createCache from "@emotion/cache";
import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";

function createEmotionCache() {
  const cache = createCache({ key: "mui", prepend: true });
  // compat helps with MUI style specificity and insertion order
  // See MUI + Emotion Next.js App Router examples
  cache.compat = true;
  return cache;
}

const lightTheme = createTheme({
  palette: { mode: "light" },
});

export default function ThemeRegistry({
  children,
}: {
  children: React.ReactNode;
}) {
  const [cache] = React.useState(() => createEmotionCache());

  useServerInsertedHTML(() => (
    <style
      data-emotion={`${cache.key} ${Object.keys(cache.inserted).join(" ")}`}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{
        __html: Object.values(cache.inserted).join(""),
      }}
    />
  ));

  return (
    <CacheProvider value={cache}>
      <ThemeProvider theme={lightTheme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </CacheProvider>
  );
}
