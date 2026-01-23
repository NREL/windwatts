// interface for model coordinates bounds
interface CoordinatesBounds {
  minLat: number;
  maxLat: number;
  minLng: number;
  maxLng: number;
}

// define constants using the CoordinatesBounds interface
export const MODEL_COORDINATES_BOUNDS: Record<string, CoordinatesBounds> = {
  "era5-quantiles": {
    minLat: 23.402,
    maxLat: 51.403,
    minLng: -137.725,
    maxLng: -44.224,
  },
  "wtk-timeseries": {
    minLat: 7.75129,
    maxLat: 78.392685,
    minLng: -179.99918,
    maxLng: 180.0,
  },
};
