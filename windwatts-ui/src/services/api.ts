import { 
  EnergyProductionRequest,
  WindspeedByLatLngRequest,
  NearestGridLocationRequest,
  WindCSVFileRequest,
  WindCSVFilesRequest
} from "../types";

export const fetchWrapper = async (url: string, options: RequestInit) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export const fetchBlobWrapper = async (url: string, options: RequestInit) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response;
};
// time_period = "global" is default for windspeed in the era5_controller. But have to include "source" param as "athena_ensemble" for ensemble model result and "athena_era5" for regular era5 result which is default anyway.
export const getWindspeedByLatLong = async ({
  lat,
  lng,
  hubHeight,
  dataModel,
  ensemble,
}: WindspeedByLatLngRequest) => {
  const url = `/api/${dataModel}/windspeed?lat=${lat}&lng=${lng}&height=${hubHeight}${ensemble ? "&ensemble=true" : ""}`;
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };
  return fetchWrapper(url, options);
};

// time_period = "all" works for era5 but have to add source param as "athena_era5" ("athena_era5" is default in era5_data_controller as source)
// time_period = "global" works for the ensemble model to fetch single global production value but have to add source as "athena_ensemble". Key is "energy_production" to get value.
export const getEnergyProduction = async ({
  lat,
  lng,
  hubHeight,
  powerCurve,
  dataModel,
  time_period = "all",
  ensemble,
}: EnergyProductionRequest) => {
  const url = `/api/${dataModel}/energy-production?lat=${lat}&lng=${lng}&height=${hubHeight}&selected_powercurve=${powerCurve}&time_period=${time_period}${ensemble ? "&ensemble=true" : ""}`;
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };
  return fetchWrapper(url, options);
};

// export const getAvailablePowerCurves = async ({
//   dataModel,
// }: { dataModel: DataModel }) => {
//   const url = `/api/${dataModel}/available-powercurves`;
//   const options = {
//     method: "GET",
//     headers: {
//       "Content-Type": "application/json",
//     },
//   };
//   return fetchWrapper(url, options);
// };

export const getAvailablePowerCurves = async () => {
  const url = `/api/era5/available-powercurves`; // if fetching from era5, replace the wtk with era5
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };
  return fetchWrapper(url, options);
};

export const getNearestGridLocation = async ({
  lat,
  lng,
  n_neighbors = 1,
  dataModel
}: NearestGridLocationRequest) => {
  const url = `/api/${dataModel}/nearest-locations?lat=${lat}&lng=${lng}&n_neighbors=${n_neighbors}`;
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };
  return fetchWrapper(url, options);
};

export const getCSVFile = async ({
  gridIndex,
  dataModel
}: WindCSVFileRequest) => {
  
  const url = `/api/${dataModel}/download-csv?gridIndex=${gridIndex}`;
  
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };

  return fetchBlobWrapper(url, options);
};

export const getBatchCSVFiles = async ({
  gridLocations,
  dataModel,
}: WindCSVFilesRequest) => {
  const url = `/api/${dataModel}/download-csv-batch`;

  const options = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ locations: gridLocations}),
  };

  return fetchBlobWrapper(url, options);
};
