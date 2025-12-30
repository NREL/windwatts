import type { DataModel } from "../types/DataModel";
import { ReadonlyURLSearchParams } from "next/navigation";

export interface UrlParams {
  lat?: number;
  lng?: number;
  zoom?: number;
  hubHeight?: number;
  powerCurve?: string;
  dataModel?: DataModel;
  ensemble?: boolean;
  lossAssumption?: number;
  partnerId?: string;
}

export const URL_PARAM_DEFAULTS = {
  zoom: 12,
  hubHeight: 40,
  powerCurve: "nrel-reference-2.5kW",
  dataModel: "era5" as DataModel,
  ensemble: false,
  lossAssumption: 0,
};

export function parseUrlParams(
  searchParams?: URLSearchParams | ReadonlyURLSearchParams
): UrlParams {
  const params =
    searchParams ||
    (typeof window !== "undefined"
      ? new URLSearchParams(window.location.search)
      : new URLSearchParams());
  const result: UrlParams = {};

  const lat = params.get("lat");
  const lng = params.get("lng");
  if (lat !== null) {
    const parsed = parseFloat(lat);
    if (!isNaN(parsed) && isFinite(parsed)) result.lat = parsed;
  }
  if (lng !== null) {
    const parsed = parseFloat(lng);
    if (!isNaN(parsed) && isFinite(parsed)) result.lng = parsed;
  }

  const zoom = params.get("zoom");
  if (zoom !== null) {
    const parsed = parseFloat(zoom);
    if (!isNaN(parsed) && parsed >= 1 && parsed <= 22)
      result.zoom = Math.round(parsed);
  }

  const hubHeight = params.get("hubHeight");
  if (hubHeight !== null) {
    const parsed = parseInt(hubHeight, 10);
    if (!isNaN(parsed) && parsed >= 10 && parsed <= 200)
      result.hubHeight = parsed;
  }

  const powerCurve = params.get("powerCurve");
  if (powerCurve) result.powerCurve = powerCurve;

  const dataModel = params.get("dataModel");
  if (dataModel === "era5" || dataModel === "wtk") result.dataModel = dataModel;

  const ensemble = params.get("ensemble");
  if (ensemble === "true") result.ensemble = true;
  else if (ensemble === "false") result.ensemble = false;

  const lossAssumption = params.get("lossAssumption");
  if (lossAssumption !== null) {
    const parsed = parseInt(lossAssumption, 10);
    if (!isNaN(parsed) && parsed >= 0 && parsed <= 100)
      result.lossAssumption = parsed;
  }

  const partnerId = params.get("partnerId");
  if (partnerId && partnerId.trim().length > 0)
    result.partnerId = partnerId.trim();

  return result;
}

export function buildUrlFromSettings(settings: {
  currentPosition: { lat: number; lng: number } | null;
  zoom: number;
  hubHeight: number;
  powerCurve: string;
  preferredModel: DataModel;
  ensemble: boolean;
  lossAssumptionPercent: number;
}): string {
  const position = settings.currentPosition;
  // Always include sane defaults when position is missing
  const lat = position?.lat ?? 39.7392;
  const lng = position?.lng ?? -104.9903;

  const params = new URLSearchParams();
  params.set("lat", lat.toFixed(4));
  params.set("lng", lng.toFixed(4));
  params.set("zoom", Math.round(settings.zoom).toString());
  params.set("hubHeight", settings.hubHeight.toString());
  params.set("powerCurve", settings.powerCurve);
  params.set("dataModel", settings.preferredModel);
  params.set("ensemble", settings.ensemble ? "true" : "false");
  params.set(
    "lossAssumption",
    String(Math.max(0, Math.min(100, settings.lossAssumptionPercent)))
  );

  const queryString = params.toString();
  return queryString ? `/?${queryString}` : "/";
}

export function hasLaunchParams(params: UrlParams): boolean {
  return (
    params.lat !== undefined &&
    params.lng !== undefined &&
    !isNaN(params.lat) &&
    !isNaN(params.lng)
  );
}
