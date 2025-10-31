"use client";

import React, { useState, useEffect, useRef, useTransition } from "react";
import { Box, IconButton } from "@mui/material";
import { Clear, Settings } from "@mui/icons-material";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useContext } from "react";
import { SettingsContext } from "../providers/SettingsContext";

export default function SearchBar() {
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const sp = useSearchParams();
  const [isPending, startTransition] = useTransition();
  const { toggleSettings } = useContext(SettingsContext);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!window.google?.maps?.places) return;
    if (!inputRef.current) return;

    autocompleteRef.current = new window.google.maps.places.Autocomplete(
      inputRef.current,
      {
        types: ["geocode", "establishment"],
        fields: ["place_id", "geometry", "formatted_address", "name"],
      }
    );

    const listener = autocompleteRef.current.addListener("place_changed", () => {
      const place = autocompleteRef.current?.getPlace();
      const loc = place?.geometry?.location;
      if (!loc) return;
      const next = new URLSearchParams(sp as any);
      next.set("lat", loc.lat().toFixed(4));
      next.set("lng", loc.lng().toFixed(4));
      setInputValue(place?.formatted_address || "");
      if (typeof window !== "undefined") {
        window.history.replaceState(null, "", `${pathname}?${next.toString()}`);
      }
      startTransition(() => router.replace(`${pathname}?${next.toString()}`));
    });

    return () => {
      if (listener) listener.remove();
    };
  }, [router, pathname, sp]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  const handleClear = () => {
    setInputValue("");
    if (inputRef.current) inputRef.current.focus();
  };

  return (
    <Box sx={{ position: "relative", width: "100%" }}>
      <input
        ref={inputRef}
        id="search-bar-input"
        type="text"
        placeholder="Enter a city, address, or landmark"
        value={inputValue}
        onChange={handleInputChange}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck="false"
        style={{
          width: "100%",
          padding: "12px 16px",
          paddingRight: inputValue ? "88px" : "56px",
          border: "1px solid #ddd",
          borderRadius: "8px",
          fontSize: "16px",
          outline: "none",
          backgroundColor: "white",
          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = "#1976d2";
          e.currentTarget.style.boxShadow =
            "0 0 0 3px rgba(25, 118, 210, 0.2), 0 2px 8px rgba(0, 0, 0, 0.1)";
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = "#ddd";
          e.currentTarget.style.boxShadow = "0 2px 8px rgba(0, 0, 0, 0.1)";
        }}
      />
      <Box sx={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", display: "flex", gap: 1 }}>
        {inputValue && (
          <IconButton
            size="small"
            onClick={handleClear}
            sx={{
              color: "#666",
              p: 1,
              minWidth: "auto",
              width: 32,
              height: 32,
              backgroundColor: "rgba(255, 255, 255, 0.9)",
              "&:hover": { backgroundColor: "rgba(255, 255, 255, 1)", color: "#333" },
            }}
          >
            <Clear sx={{ fontSize: 18 }} />
          </IconButton>
        )}
        <IconButton
          size="small"
          onClick={toggleSettings}
          sx={{
            color: "#666",
            p: 1,
            minWidth: "auto",
            width: 32,
            height: 32,
            backgroundColor: "rgba(255, 255, 255, 0.9)",
            "&:hover": { backgroundColor: "rgba(255, 255, 255, 1)", color: "#333" },
          }}
          aria-label="Open settings"
        >
          <Settings sx={{ fontSize: 18 }} />
        </IconButton>
      </Box>
    </Box>
  );
}
