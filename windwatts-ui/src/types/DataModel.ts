export type DataModel = "wtk" | "era5";

export type DataModelInfo = {
  label: string;
  source_href: string;
  help_href: string;
  description: string;
  year_range: string;
  wind_speed_heights: string[];
  wind_direction_heights: string[];
};
