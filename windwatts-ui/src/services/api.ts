import {
  EnergyProductionRequest,
  WindspeedByLatLngRequest,
  NearestGridLocationRequest,
  WindCSVFileRequest,
  WindCSVFilesRequest,
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
// V1 API: Uses unified endpoint with model as path parameter
export const getWindspeedByLatLong = async ({
  lat,
  lng,
  hubHeight,
  dataModel,
}: WindspeedByLatLngRequest) => {
  const url = `/api/v1/${dataModel}/windspeed?lat=${lat}&lng=${lng}&height=${hubHeight}`;
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };
  return fetchWrapper(url, options);
};

// V1 API:
// period options: "all" (default), "summary", "annual", "monthly" (varies by model)
export const getEnergyProduction = async ({
  lat,
  lng,
  hubHeight,
  turbine,
  dataModel,
  period = "all",
}: EnergyProductionRequest) => {
  const url = `/api/v1/${dataModel}/production?lat=${lat}&lng=${lng}&height=${hubHeight}&turbine=${turbine}&period=${period}`;
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

// V1 API: Power curves are model-agnostic
export const getAvailablePowerCurves = async () => {
  const url = `/api/v1/powercurves`;
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };
  return fetchWrapper(url, options);
};

// V1 API: Grid points lookup
export const getNearestGridLocation = async ({
  lat,
  lng,
  n_neighbors = 1,
  dataModel,
}: NearestGridLocationRequest) => {
  const url = `/api/v1/${dataModel}/grid-points?lat=${lat}&lng=${lng}&limit=${n_neighbors}`;
  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };
  return fetchWrapper(url, options);
};

// V1 API: Single timeseries CSV download (default source is s3)
export const getCSVFile = async ({
  gridIndex,
  dataModel,
}: WindCSVFileRequest) => {
  const url = `/api/v1/${dataModel}/timeseries?gridIndex=${gridIndex}`;

  const options = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };

  return fetchBlobWrapper(url, options);
};

// V1 API: Batch timeseries CSV download as ZIP
export const getBatchCSVFiles = async ({
  gridLocations,
  dataModel,
}: WindCSVFilesRequest) => {
  const url = `/api/v1/${dataModel}/timeseries/batch`;

  const options = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      locations: gridLocations,
    }),
  };

  return fetchBlobWrapper(url, options);
};
