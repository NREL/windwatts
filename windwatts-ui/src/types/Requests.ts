import { DataModel } from "./DataModel";
import { GridLocation } from "./GridLocation";

export interface WindspeedByLatLngRequest {
  lat: number;
  lng: number;
  hubHeight: number;
  dataModel: DataModel;
}

export interface EnergyProductionRequest {
  lat: number;
  lng: number;
  hubHeight: number;
  turbine: string;
  dataModel: DataModel;
  period?: string;
}

export interface NearestGridLocationRequest {
  lat: number;
  lng: number;
  n_neighbors?: number;
  dataModel: DataModel;
}

export interface WindCSVFileRequest {
  gridIndex: string;
  dataModel: DataModel;
}

export interface WindCSVFilesRequest {
  gridLocations: GridLocation[];
  dataModel: string;
}
