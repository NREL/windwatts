"use client";

import { GoogleMap, Marker, useLoadScript } from "@react-google-maps/api";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useState, ComponentType } from "react";

const containerStyle = { width: "100%", height: "400px" };

// Cast to generic component types to satisfy TS in this environment
const GoogleMapAny = GoogleMap as unknown as ComponentType<any>;
const MarkerAny = Marker as unknown as ComponentType<any>;

export default function Map() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const lat = Number(searchParams.get("lat") || 39.7392);
  const lng = Number(searchParams.get("lng") || -104.9903);

  const [center, setCenter] = useState({ lat, lng });

  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: process.env.NEXT_PUBLIC_MAP_API_KEY || "",
    libraries: ["places", "marker"],
    version: "quarterly"
  });

  const onClick = useCallback((e: google.maps.MapMouseEvent) => {
    if (!e.latLng) return;
    const next = new URLSearchParams(searchParams as any);
    next.set("lat", e.latLng.lat().toFixed(4));
    next.set("lng", e.latLng.lng().toFixed(4));
    router.replace(`${pathname}?${next.toString()}`);
    setCenter({ lat: e.latLng.lat(), lng: e.latLng.lng() });
  }, [router, pathname, searchParams]);

  if (loadError) return null;
  if (!isLoaded) return null;

  return (
    <GoogleMapAny
      mapContainerStyle={containerStyle}
      center={center}
      zoom={10}
      onClick={onClick}
      options={{ mapId: process.env.NEXT_PUBLIC_MAP_ID }}
    >
      <MarkerAny position={center} />
    </GoogleMapAny>
  );
}
