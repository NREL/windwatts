export const DATA_MODELS = [
  "wtk-timeseries",
  "era5-quantiles",
  "ensemble-quantiles",
  "era5-timeseries",
] as const;

export type DataModel = typeof DATA_MODELS[number];

export type DataModelInfo = {
  label: string;
  source_href: string;
  help_href: string;
  description: string;
  year_range: string;
  wind_speed_heights: string[];
  wind_direction_heights: string[];
};
