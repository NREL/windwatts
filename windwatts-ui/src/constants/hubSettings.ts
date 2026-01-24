import { DataModel, Heights } from "../types";

export const HUB_HEIGHTS: Record<DataModel | "default", Heights> = {
  "era5-quantiles": { values: [30, 40, 50, 60, 80, 100], interpolation: null },
  "wtk-timeseries": { values: [40, 60, 80, 100, 120, 140], interpolation: 10 },
  "ensemble-quantiles": { values: [30, 40, 50, 60, 80, 100], interpolation: null },
  "era5-timeseries": { values: [30, 40, 50, 60, 80, 100], interpolation: null },
  default: { values: [40, 60, 80, 100], interpolation: null },
};
