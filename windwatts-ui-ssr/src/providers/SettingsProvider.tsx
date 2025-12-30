"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  SettingsContext,
  defaultValues,
  type CurrentPosition,
  type StoredSettings,
} from "./SettingsContext";
import type { DataModel } from "../types/DataModel";
import {
  parseUrlParams,
  hasLaunchParams,
  URL_PARAM_DEFAULTS,
} from "../utils/urlParams";

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const [uiState, setUiState] = useState({
    settingsOpen: false,
    resultsOpen: false,
  });

  const toggleSettings = useCallback(() => {
    setUiState((prev) => ({ ...prev, settingsOpen: !prev.settingsOpen }));
  }, []);

  const toggleResults = useCallback(() => {
    setUiState((prev) => ({ ...prev, resultsOpen: !prev.resultsOpen }));
  }, []);

  // Derive state directly from searchParams to avoid synchronization loops
  const params = useMemo(() => parseUrlParams(searchParams), [searchParams]);
  const hasPos = hasLaunchParams(params);

  const currentPosition = useMemo(
    () =>
      hasPos && params.lat && params.lng
        ? { lat: params.lat, lng: params.lng }
        : { lat: 39.7392, lng: -104.9903 },
    [hasPos, params.lat, params.lng]
  );

  const zoom = params.zoom ?? URL_PARAM_DEFAULTS.zoom;
  const hubHeight = params.hubHeight ?? URL_PARAM_DEFAULTS.hubHeight;
  const powerCurve = params.powerCurve ?? URL_PARAM_DEFAULTS.powerCurve;
  const preferredModel = params.dataModel ?? URL_PARAM_DEFAULTS.dataModel;
  const ensemble = params.ensemble ?? URL_PARAM_DEFAULTS.ensemble;
  const lossAssumptionFactor =
    1 - (params.lossAssumption ?? URL_PARAM_DEFAULTS.lossAssumption) / 100;

  const updateUrl = useCallback(
    (updates: Record<string, string>) => {
      const next = new URLSearchParams(searchParams as any);
      Object.entries(updates).forEach(([key, value]) => {
        next.set(key, value);
      });
      router.replace(`${pathname}?${next.toString()}`);
    },
    [router, pathname, searchParams]
  );

  const setCurrentPosition = useCallback(
    (
      position:
        | CurrentPosition
        | null
        | ((prev: CurrentPosition | null) => CurrentPosition | null)
    ) => {
      let newPos: CurrentPosition | null;
      if (typeof position === "function") {
        newPos = position(currentPosition);
      } else {
        newPos = position;
      }

      if (newPos) {
        updateUrl({
          lat: newPos.lat.toFixed(4),
          lng: newPos.lng.toFixed(4),
        });
      }
    },
    [currentPosition, updateUrl]
  );

  const setZoom = useCallback(
    (newZoom: number) => {
      updateUrl({ zoom: Math.round(newZoom).toString() });
    },
    [updateUrl]
  );

  const setHubHeight = useCallback(
    (height: number) => {
      updateUrl({ hubHeight: String(height) });
    },
    [updateUrl]
  );

  const setPowerCurve = useCallback(
    (curve: string) => {
      updateUrl({ powerCurve: curve });
    },
    [updateUrl]
  );

  const setPreferredModel = useCallback(
    (model: DataModel) => {
      updateUrl({ dataModel: model });
    },
    [updateUrl]
  );

  const setEnsemble = useCallback(
    (newEnsemble: boolean) => {
      updateUrl({ ensemble: newEnsemble ? "true" : "false" });
    },
    [updateUrl]
  );

  const setLossAssumptionFactor = useCallback(
    (factor: number) => {
      const clamped = Math.max(0, Math.min(1, Number(factor)));
      const val = Math.round((1 - clamped) * 100);
      updateUrl({ lossAssumption: String(val) });
    },
    [updateUrl]
  );

  const setLossAssumptionPercent = useCallback(
    (percent: number) => {
      const num = Math.max(0, Math.min(100, Number(percent)));
      updateUrl({ lossAssumption: String(num) });
    },
    [updateUrl]
  );

  const contextValue = useMemo(
    () => ({
      settingsOpen: uiState.settingsOpen,
      toggleSettings,
      resultsOpen: uiState.resultsOpen,
      toggleResults,
      currentPosition,
      setCurrentPosition,
      zoom,
      setZoom,
      hubHeight,
      setHubHeight,
      powerCurve,
      setPowerCurve,
      preferredModel,
      setPreferredModel,
      ensemble,
      setEnsemble,
      lossAssumptionFactor,
      lossAssumptionPercent: Math.round((1 - lossAssumptionFactor) * 100),
      setLossAssumptionFactor,
      setLossAssumptionPercent,
    }),
    [
      uiState.settingsOpen,
      toggleSettings,
      uiState.resultsOpen,
      toggleResults,
      currentPosition,
      setCurrentPosition,
      zoom,
      setZoom,
      hubHeight,
      setHubHeight,
      powerCurve,
      setPowerCurve,
      preferredModel,
      setPreferredModel,
      ensemble,
      setEnsemble,
      lossAssumptionFactor,
      setLossAssumptionFactor,
      setLossAssumptionPercent,
    ]
  );

  return (
    <SettingsContext.Provider value={contextValue}>
      {children}
    </SettingsContext.Provider>
  );
}
