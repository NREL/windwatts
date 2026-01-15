export interface TurbineInfo {
  label: string; // Displayed Label
  minHeight: number; // Preferred min height
  maxHeight: number; // Preferred max height
}

export const TURBINE_DATA: Record<string, TurbineInfo> = {
  "nlr-reference-2.5kW": {
    label: "NLR Reference 2.5kW",
    minHeight: 20,
    maxHeight: 40,
  },
  "nlr-reference-100kW": {
    label: "NLR Reference 100kW",
    minHeight: 30,
    maxHeight: 80,
  },
  "nlr-reference-250kW": {
    label: "NLR Reference 250kW",
    minHeight: 40,
    maxHeight: 100,
  },
  "nlr-reference-2000kW": {
    label: "NLR Reference 2000kW",
    minHeight: 60,
    maxHeight: 140,
  },
  "bergey-excel-15": {
    label: "Bergey Excel 15kW",
    minHeight: 20,
    maxHeight: 50,
  },
  "eocycle-25": {
    label: "Eocycle 25kW",
    minHeight: 25,
    maxHeight: 60,
  },
  "northern-100": {
    label: "Northern Power 100kW",
    minHeight: 30,
    maxHeight: 80,
  },
  siva_250kW_30m_rotor_diameter: {
    label: "Siva 250kW (30m rotor diameter)",
    minHeight: 40,
    maxHeight: 100,
  },
  siva_250kW_32m_rotor_diameter: {
    label: "Siva 250kW (32m rotor diameter)",
    minHeight: 40,
    maxHeight: 100,
  },
  siva_750_u50: {
    label: "Siva 750kW (50m rotor diameter)",
    minHeight: 50,
    maxHeight: 120,
  },
  siva_750_u57: {
    label: "Siva 750kW (57m rotor diameter)",
    minHeight: 50,
    maxHeight: 140,
  },
};

export const POWER_CURVE_LABEL: Record<string, string> = Object.entries(
  TURBINE_DATA
).reduce(
  (acc, [key, turbine]) => {
    acc[key] = turbine.label;
    return acc;
  },
  {} as Record<string, string>
);

export const VALID_POWER_CURVES = Object.keys(TURBINE_DATA);