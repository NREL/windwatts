import { createContext } from "react";

export interface StoredUnits {
  windspeed: string;
  output: string;
}

export interface Units {
  units: StoredUnits;
  setUnits: (units: StoredUnits) => void;
  updateUnit: (key: string, value: string) => void;
  updateUnits: (newValues: StoredUnits) => void;
}

export const defaultUnitValues: StoredUnits = {
  windspeed: "mph",
  output: "kWh",
};

export const defaultValues: Units = {
  units: defaultUnitValues,
  setUnits: () => {},
  updateUnit: () => {},
  updateUnits: () => {},
};

export const UnitsContext = createContext<Units>(defaultValues);
