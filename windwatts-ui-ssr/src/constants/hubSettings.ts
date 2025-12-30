import type { DataModel } from "../types/DataModel";

export type Heights = { values: number[]; interpolation: number | null };

export const HUB_HEIGHTS: Record<DataModel | "default", Heights> = {
  era5: { values: [30, 40, 50, 60, 80, 100], interpolation: null },
  wtk: { values: [40, 60, 80, 100, 120, 140], interpolation: 10 },
  default: { values: [40, 60, 80, 100], interpolation: null }
};
