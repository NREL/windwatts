import { DataModelInfo } from "../types";

export const DATA_MODEL_INFO: Record<string, DataModelInfo> = {
  era5: {
    label: "ERA5",
    source_href:
      "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5",
    help_href:
      "https://github.com/NREL/dw-tap-api/blob/main/docs/about/era5.md",
    description: "ERA5 (ECMWF Reanalysis v5) Dataset",
    year_range: "2020-2023",
    wind_speed_heights: ["10m", "30m", "40m", "50m", "60m", "80m", "100m"],
    wind_direction_heights: ["10m", "100m"],
  },
  wtk: {
    label: "NREL's 20-year WTK-LED dataset",
    source_href:
      "https://www.energy.gov/eere/wind/articles/new-wind-resource-database-includes-updated-wind-toolkit",
    help_href: "",
    description: "NREL's 20-year WTK-LED Dataset",
    year_range: "",
    wind_speed_heights: [],
    wind_direction_heights: [],
  },
};
