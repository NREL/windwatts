import { createContext } from "react";
import type { DataModel } from "../types/DataModel";

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
  ensemble: boolean;
  lossAssumptionFactor: number;
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
  setEnsemble: (ensemble: boolean) => void;
  setLossAssumptionFactor: (factor: number) => void;
  setLossAssumptionPercent: (percent: number) => void;
}

export const defaultValues: StoredSettings = {
  settingsOpen: false,
  resultsOpen: false,
  currentPosition: null,
  zoom: 12,
  hubHeight: 40,
  powerCurve: "nrel-reference-2.5kW",
  preferredModel: "era5",
  ensemble: false,
  lossAssumptionFactor: 1.0,
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
  setEnsemble: () => {},
  setLossAssumptionFactor: () => {},
  setLossAssumptionPercent: () => {},
});
