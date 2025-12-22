import { createContext } from "react";
import { DataModel } from "../types";

export interface CurrentPosition {
  lat: number;
  lng: number;
}

export interface StoredSettings {
  settingsOpen: boolean;
  resultsOpen: boolean;
  currentPosition: CurrentPosition | null;
  zoom: number;
  hubHeight: number;
  powerCurve: string;
  preferredModel: DataModel;
  lossAssumptionFactor: number;
  windspeedUnit: string;
}

export interface Settings extends StoredSettings {
  lossAssumptionPercent: number;
  toggleSettings: () => void;
  toggleResults: () => void;
  setCurrentPosition: (
    position:
      | CurrentPosition
      | null
      | ((prev: CurrentPosition | null) => CurrentPosition | null)
  ) => void;
  setZoom: (zoom: number) => void;
  setHubHeight: (hubHeight: number) => void;
  setPowerCurve: (curve: string) => void;
  setPreferredModel: (preferredModel: DataModel) => void;
  setLossAssumptionFactor: (factor: number) => void;
  setLossAssumptionPercent: (percent: number) => void;
  setWindspeedUnit: (unit: string) => void;
}

export const defaultValues: StoredSettings = {
  settingsOpen: false,
  resultsOpen: false,
  currentPosition: null,
  zoom: 12,
  hubHeight: 40,
  powerCurve: "nlr-reference-100kW",
  preferredModel: "era5",
  lossAssumptionFactor: 1.0, // 1.0 = 0% loss, 0.83 = 17% loss defaulted to 0% loss
  windspeedUnit: "mph",
};

export const SettingsContext = createContext<Settings>({
  ...defaultValues,
  lossAssumptionPercent: 0,
  toggleSettings: () => {},
  toggleResults: () => {},
  setCurrentPosition: () => {},
  setZoom: () => {},
  setHubHeight: () => {},
  setPowerCurve: () => {},
  setPreferredModel: () => {},
  setLossAssumptionFactor: () => {},
  setLossAssumptionPercent: () => {},
  setWindspeedUnit: () => {},
});
