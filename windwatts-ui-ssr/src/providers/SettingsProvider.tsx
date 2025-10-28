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
  buildUrlFromSettings,
  parseUrlParams,
  hasLaunchParams,
  URL_PARAM_DEFAULTS,
} from "../utils/urlParams";

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const [settings, setSettings] = useState<StoredSettings>(() => {
    const params = parseUrlParams(searchParams);
    const hasPos = hasLaunchParams(params);
    return {
      ...defaultValues,
      currentPosition:
        hasPos && params.lat && params.lng
          ? { lat: params.lat, lng: params.lng }
          : { lat: 39.7392, lng: -104.9903 },
      zoom: params.zoom ?? URL_PARAM_DEFAULTS.zoom,
      hubHeight: params.hubHeight ?? URL_PARAM_DEFAULTS.hubHeight,
      powerCurve: params.powerCurve ?? URL_PARAM_DEFAULTS.powerCurve,
      preferredModel: params.dataModel ?? URL_PARAM_DEFAULTS.dataModel,
      ensemble: params.ensemble ?? URL_PARAM_DEFAULTS.ensemble,
      lossAssumptionFactor:
        1 - (params.lossAssumption ?? URL_PARAM_DEFAULTS.lossAssumption) / 100,
    };
  });

  useEffect(() => {
    const url = buildUrlFromSettings({
      currentPosition: settings.currentPosition,
      zoom: settings.zoom,
      hubHeight: settings.hubHeight,
      powerCurve: settings.powerCurve,
      preferredModel: settings.preferredModel,
      ensemble: settings.ensemble,
      lossAssumptionPercent: Math.round(
        (1 - settings.lossAssumptionFactor) * 100
      ),
    });

    const current = `${pathname}${searchParams?.toString() ? `?${searchParams.toString()}` : ""}`;
    if (url !== current) router.replace(url);
  }, [router, pathname, searchParams, settings]);

  const setCurrentPosition = useCallback(
    (
      position:
        | CurrentPosition
        | null
        | ((prev: CurrentPosition | null) => CurrentPosition | null)
    ) => {
      setSettings((current) => ({
        ...current,
        currentPosition:
          typeof position === "function"
            ? position(current.currentPosition)
            : position || { lat: 39.7392, lng: -104.9903 },
      }));
    },
    []
  );

  const setZoom = useCallback((zoom: number) => {
    setSettings((current) => ({ ...current, zoom: Math.round(zoom) }));
  }, []);

  const setHubHeight = useCallback((height: number) => {
    setSettings((current) => ({ ...current, hubHeight: height }));
  }, []);

  const setPowerCurve = useCallback((curve: string) => {
    setSettings((current) => ({ ...current, powerCurve: curve }));
  }, []);

  const setPreferredModel = useCallback((model: DataModel) => {
    setSettings((current) => ({ ...current, preferredModel: model }));
  }, []);

  const setEnsemble = useCallback((ensemble: boolean) => {
    setSettings((current) => ({ ...current, ensemble }));
  }, []);

  const setLossAssumptionFactor = useCallback((factor: number) => {
    const clamped = Math.max(0, Math.min(1, Number(factor)));
    setSettings((current) => ({ ...current, lossAssumptionFactor: clamped }));
  }, []);

  const setLossAssumptionPercent = useCallback(
    (percent: number) => {
      const num = Math.max(0, Math.min(100, Number(percent)));
      const factor = (100 - num) / 100;
      setLossAssumptionFactor(factor);
    },
    [setLossAssumptionFactor]
  );

  const toggleSettings = useCallback(() => {
    setSettings((current) => ({
      ...current,
      settingsOpen: !current.settingsOpen,
    }));
  }, []);

  const toggleResults = useCallback(() => {
    setSettings((current) => ({
      ...current,
      resultsOpen: !current.resultsOpen,
    }));
  }, []);

  const contextValue = useMemo(
    () => ({
      settingsOpen: settings.settingsOpen,
      toggleSettings,
      resultsOpen: settings.resultsOpen,
      toggleResults,
      currentPosition: settings.currentPosition,
      setCurrentPosition,
      zoom: settings.zoom,
      setZoom,
      hubHeight: settings.hubHeight,
      setHubHeight,
      powerCurve: settings.powerCurve,
      setPowerCurve,
      preferredModel: settings.preferredModel,
      setPreferredModel,
      ensemble: settings.ensemble,
      setEnsemble,
      lossAssumptionFactor:
        settings.lossAssumptionFactor ?? defaultValues.lossAssumptionFactor,
      lossAssumptionPercent: Math.round(
        (1 -
          (settings.lossAssumptionFactor ??
            defaultValues.lossAssumptionFactor)) *
        100
      ),
      setLossAssumptionFactor,
      setLossAssumptionPercent,
    }),
    [
      settings,
      setCurrentPosition,
      setZoom,
      setHubHeight,
      setPowerCurve,
      setPreferredModel,
      setEnsemble,
      setLossAssumptionFactor,
      setLossAssumptionPercent,
      toggleSettings,
      toggleResults,
    ]
  );

  return (
    <SettingsContext.Provider value={contextValue}>
      {children}
    </SettingsContext.Provider>
  );
}
