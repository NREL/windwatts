"use client";

import { Autocomplete, useLoadScript } from "@react-google-maps/api";
import { TextField, InputAdornment } from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useRef, useState, useTransition } from "react";

export default function SearchBar() {
  const router = useRouter();
  const pathname = usePathname();
  const sp = useSearchParams();
  const [value, setValue] = useState("");
  const [isPending, startTransition] = useTransition();
  const acRef = useRef<google.maps.places.Autocomplete | null>(null);

  const { isLoaded } = useLoadScript({
    googleMapsApiKey: process.env.NEXT_PUBLIC_MAP_API_KEY || "",
    libraries: ["places"],
    version: "quarterly"
  });

  const onLoad = (ac: google.maps.places.Autocomplete) => {
    acRef.current = ac;
  };

  const onPlaceChanged = () => {
    const ac = acRef.current;
    if (!ac) return;
    const place = ac.getPlace();
    const loc = place?.geometry?.location;
    if (!loc) return;
    const next = new URLSearchParams(sp as any);
    next.set("lat", loc.lat().toFixed(4));
    next.set("lng", loc.lng().toFixed(4));
    if (typeof window !== "undefined") {
      window.history.replaceState(null, "", `${pathname}?${next.toString()}`);
    }
    startTransition(() => router.replace(`${pathname}?${next.toString()}`));
  };

  if (!isLoaded) return null;

  return (
    <Autocomplete onLoad={onLoad} onPlaceChanged={onPlaceChanged}>
      <TextField
        placeholder="Enter a city, address, or landmark"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" />
            </InputAdornment>
          )
        }}
        fullWidth
      />
    </Autocomplete>
  );
}
