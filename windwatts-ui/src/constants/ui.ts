export const SEARCH_MIN_LENGTH = 2;
export const SEARCH_DEBOUNCE_DELAY = 300;

export const TRANSITION_DURATION = 300;
export const FLAG_RESET_DELAY = 100;

export const MOBILE_BOTTOM_SHEET_HEIGHT = "90vh";
export const MOBILE_BOTTOM_SHEET_COLLAPSED_HEIGHT = "120px";
export const SETTINGS_MODAL_WIDTH = 400;
export const SETTINGS_MODAL_MAX_HEIGHT = "80vh";

export const DEFAULT_MAP_CENTER = { lat: 39.7392, lng: -104.9903 };
export const INITIAL_MAP_ZOOM = 12;

export const MOBILE_BREAKPOINT = "sm";

export const MONTH_NAMES = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

export const GEOLOCATION_OPTIONS = {
  maximumAge: 60000,
  timeout: 5000,
  enableHighAccuracy: true,
};

export const OUT_OF_BOUNDS_MARKER_CONFIG: google.maps.Symbol | undefined =
  typeof window !== "undefined" && window.google?.maps
    ? {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 10,
        fillColor: "#d32f2f",
        fillOpacity: 1,
        strokeWeight: 2,
        strokeColor: "#fff",
      }
    : undefined;

export const SNACKBAR_DURATION_SUCCESS = 3000;
export const SNACKBAR_DURATION_ERROR = 4000;
