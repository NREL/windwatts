import { DataModel } from "../types";
import { MODEL_COORDINATES_BOUNDS, VALID_POWER_CURVES } from "../constants";

export interface UrlParams {
  lat?: number;
  lng?: number;
  zoom?: number;
  hubHeight?: number;
  powerCurve?: string;
  dataModel?: DataModel;
  lossAssumption?: number;
  partnerId?: string;
  windspeedUnit?: string;
}

export const URL_PARAM_DEFAULTS = {
  zoom: 12,
  hubHeight: 40,
  powerCurve: "nlr-reference-100kW",
  dataModel: "era5" as DataModel,
  lossAssumption: 0,
  windspeedUnit: "mph",
};

export function parseUrlParams(searchParams?: URLSearchParams): UrlParams {
  const params = searchParams || new URLSearchParams(window.location.search);
  const result: UrlParams = {};

  const lat = params.get("lat");
  const lng = params.get("lng");

  if (lat !== null) {
    const parsed = parseFloat(lat);
    if (!isNaN(parsed) && isFinite(parsed)) {
      result.lat = parsed;
    }
  }

  if (lng !== null) {
    const parsed = parseFloat(lng);
    if (!isNaN(parsed) && isFinite(parsed)) {
      result.lng = parsed;
    }
  }

  const zoom = params.get("zoom");
  if (zoom !== null) {
    const parsed = parseFloat(zoom);
    if (!isNaN(parsed) && parsed >= 1 && parsed <= 22) {
      result.zoom = Math.round(parsed);
    }
  }

  const hubHeight = params.get("hubHeight");
  if (hubHeight !== null) {
    const parsed = parseInt(hubHeight, 10);
    if (!isNaN(parsed) && parsed >= 10 && parsed <= 200) {
      result.hubHeight = parsed;
    }
  }

  const powerCurve = params.get("powerCurve");
  if (powerCurve && VALID_POWER_CURVES.includes(powerCurve)) {
    result.powerCurve = powerCurve;
  }

  const dataModel = params.get("dataModel");
  if (dataModel === "era5" || dataModel === "wtk" || dataModel === "ensemble") {
    result.dataModel = dataModel;
  }

  const lossAssumption = params.get("lossAssumption");
  if (lossAssumption !== null) {
    const parsed = parseInt(lossAssumption, 10);
    if (!isNaN(parsed) && parsed >= 0 && parsed <= 100) {
      result.lossAssumption = parsed;
    }
  }

  const partnerId = params.get("partnerId");
  if (partnerId && partnerId.trim().length > 0) {
    result.partnerId = partnerId.trim();
  }

  const windspeedUnit = params.get("windspeedUnit");
  if (windspeedUnit === "mph") {
    result.windspeedUnit = "mph";
  } else if (windspeedUnit === "ms") {
    result.windspeedUnit = "m/s";
  }

  return result;
}

export function buildUrlFromSettings(settings: {
  currentPosition: { lat: number; lng: number } | null;
  zoom: number;
  hubHeight: number;
  powerCurve: string;
  preferredModel: DataModel;
  lossAssumptionPercent: number;
  windspeedUnit: string;
}): string {
  const position = settings.currentPosition;
  if (
    !position ||
    typeof position.lat !== "number" ||
    typeof position.lng !== "number"
  ) {
    return "/";
  }

  const params = new URLSearchParams();

  params.set("lat", position.lat.toFixed(4));
  params.set("lng", position.lng.toFixed(4));

  if (settings.hubHeight !== URL_PARAM_DEFAULTS.hubHeight) {
    params.set("hubHeight", settings.hubHeight.toString());
  }

  if (settings.powerCurve !== URL_PARAM_DEFAULTS.powerCurve) {
    params.set("powerCurve", settings.powerCurve);
  }

  if (settings.preferredModel !== URL_PARAM_DEFAULTS.dataModel) {
    params.set("dataModel", settings.preferredModel);
  }

  if (settings.lossAssumptionPercent > 0) {
    params.set("lossAssumption", settings.lossAssumptionPercent.toString());
  }

  if (settings.zoom !== URL_PARAM_DEFAULTS.zoom) {
    params.set("zoom", Math.round(settings.zoom).toString());
  }

  if (
    settings.windspeedUnit &&
    settings.windspeedUnit !== URL_PARAM_DEFAULTS.windspeedUnit
  ) {
    // Convert "m/s" to "ms" for URL compatibility
    const urlValue =
      settings.windspeedUnit === "m/s" ? "ms" : settings.windspeedUnit;
    params.set("windspeedUnit", urlValue);
  }

  const queryString = params.toString();
  return queryString ? `/?${queryString}` : "/";
}

export function validateCoordinates(
  lat: number,
  lng: number,
  model: DataModel
): boolean {
  const bounds = MODEL_COORDINATES_BOUNDS[model];
  if (!bounds) return false;

  return (
    lat >= bounds.minLat &&
    lat <= bounds.maxLat &&
    lng >= bounds.minLng &&
    lng <= bounds.maxLng
  );
}

export function hasLaunchParams(params: UrlParams): boolean {
  return (
    params.lat !== undefined &&
    params.lng !== undefined &&
    !isNaN(params.lat) &&
    !isNaN(params.lng)
  );
}
