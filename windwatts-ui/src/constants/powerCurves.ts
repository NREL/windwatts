export const POWER_CURVE_LABEL: Record<string, string> = {
  "nlr-reference-2.5kW": "NLR Reference 2.5kW",
  "nlr-reference-100kW": "NLR Reference 100kW",
  "nlr-reference-250kW": "NLR Reference 250kW",
  "nlr-reference-2000kW": "NLR Reference 2000kW",
  "bergey-excel-15": "Bergey Excel 15kW",
  "eocycle-25": "Eocycle 25kW",
  "northern-100": "Northern Power 100kW",
  siva_250kW_30m_rotor_diameter: "Siva 250kW (30m rotor diameter)",
  siva_250kW_32m_rotor_diameter: "Siva 250kW (32m rotor diameter)",
  siva_750_u50: "Siva 750kW (50m rotor diameter)",
  siva_750_u57: "Siva 750kW (57m rotor diameter)",
};

export const VALID_POWER_CURVES = Object.keys(POWER_CURVE_LABEL);
