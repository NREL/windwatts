export function getApiBase(): string {
  if (process.env.WINDWATTS_API_BASE) return process.env.WINDWATTS_API_BASE;
  if (process.env.DOCKER === "1") return "http://nginx/api";
  return "http://localhost:8080/api";
}

export async function fetchJson<T>(
  pathOrUrl: string,
  init?: RequestInit
): Promise<T> {
  const url = pathOrUrl.startsWith("http")
    ? pathOrUrl
    : `${getApiBase()}${pathOrUrl}`;
  const res = await fetch(url, { cache: "no-store", ...(init || {}) });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Fetch failed ${res.status} ${res.statusText}: ${url} ${text}`
    );
  }
  return res.json() as Promise<T>;
}

export const era5 = {
  async windspeed(params: {
    lat: number;
    lng: number;
    height: number;
    source?: string;
    avg_type?: string;
    ensemble?: boolean;
  }) {
    const url = new URL(`${getApiBase()}/era5/windspeed`);
    url.searchParams.set("lat", String(params.lat));
    url.searchParams.set("lng", String(params.lng));
    url.searchParams.set("height", String(params.height));
    if (params.source) url.searchParams.set("source", params.source);
    if (params.avg_type) url.searchParams.set("avg_type", params.avg_type);
    if (params.ensemble) url.searchParams.set("ensemble", "true");
    return fetchJson<any>(url.toString());
  },
  async energyProduction(params: {
    lat: number;
    lng: number;
    height: number;
    powerCurve: string;
    time_period?: string;
    source?: string;
    ensemble?: boolean;
  }) {
    const time = params.time_period || "global";
    const url = new URL(`${getApiBase()}/era5/energy-production/${time}`);
    url.searchParams.set("lat", String(params.lat));
    url.searchParams.set("lng", String(params.lng));
    url.searchParams.set("height", String(params.height));
    url.searchParams.set("selected_powercurve", params.powerCurve);
    if (params.source) url.searchParams.set("source", params.source);
    if (params.ensemble) url.searchParams.set("ensemble", "true");
    return fetchJson<any>(url.toString());
  },
  async availablePowerCurves() {
    const url = `${getApiBase()}/era5/available-powercurves`;
    return fetchJson<{ available_power_curves: string[] }>(url);
  },
};
