"use client";

import { GoogleMap, Marker } from "@react-google-maps/api";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState, ComponentType, useTransition } from "react";

// Cast to generic component types to satisfy TS in this environment
const GoogleMapAny = GoogleMap as unknown as ComponentType<any>;
const MarkerAny = Marker as unknown as ComponentType<any>;

export default function Map({ height = "100%" }: { height?: string | number }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [isPending, startTransition] = useTransition();

  const lat = Number(searchParams.get("lat") || 39.7392);
  const lng = Number(searchParams.get("lng") || -104.9903);

  const [center, setCenter] = useState({ lat, lng });

  // Keep local center in sync when URL changes externally
  useEffect(() => {
    setCenter({ lat, lng });
  }, [lat, lng]);

  const onClick = useCallback((e: any) => {
    if (!e?.latLng) return;
    const next = new URLSearchParams(searchParams as any);
    const newLat = e.latLng.lat();
    const newLng = e.latLng.lng();
    next.set("lat", newLat.toFixed(4));
    next.set("lng", newLng.toFixed(4));

    // Optimistically update UI and URL immediately
    setCenter({ lat: newLat, lng: newLng });
    if (typeof window !== "undefined") {
      window.history.replaceState(null, "", `${pathname}?${next.toString()}`);
    }

    // Trigger Next navigation without blocking UI
    startTransition(() => {
      router.replace(`${pathname}?${next.toString()}`);
    });
  }, [router, pathname, searchParams]);

  return (
    <div style={{ width: "100%", height }}>
      <GoogleMapAny
        mapContainerStyle={{ width: "100%", height: "100%" }}
        center={center}
        zoom={10}
        onClick={onClick}
        options={{
          mapId: process.env.NEXT_PUBLIC_MAP_ID,
          clickableIcons: true,
          disableDefaultUI: false,
          zoomControl: true,
          streetViewControl: false,
          fullscreenControl: false
        }}
      >
        <MarkerAny position={center} />
      </GoogleMapAny>
    </div>
  );
}
