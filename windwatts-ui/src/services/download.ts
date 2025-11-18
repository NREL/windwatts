import { formatCoordinate } from "../utils";

export const downloadBlobAsFile = (blob: Blob, filename: string) => {
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(downloadUrl);
};

export const generateWindDataFilename = (gridLat: number, gridLng: number) =>{
    if (gridLat === undefined || gridLng === undefined){
        return `wind_data_${Date.now()}.csv`; // fallback with timestamp
    }
    return `wind_data_${formatCoordinate(gridLat)}_${formatCoordinate(gridLng)}.csv`
}

export const downloadWindDataCSV = async (
  response: Response,
  gridLat: number,
  gridLng: number
) => {
  try {
    const blob = await response.blob();
    const filename = generateWindDataFilename(gridLat, gridLng);
    downloadBlobAsFile(blob, filename);
    return { success: true, filename };
  } catch (error) {
    console.error('Failed to process download:', error);
    throw error;
  }
};

export const downloadWindDataZIP = async (
  response: Response,
  lat: number,
  lng: number
) => {
  try {
    const blob = await response.blob();
    const filename = `wind_data_${formatCoordinate(lat)}_${formatCoordinate(lng)}_neighbors.zip`;
    downloadBlobAsFile(blob, filename);
    return { success: true, filename };
  } catch (error) {
    console.error('Failed to process ZIP download:', error);
    throw error;
  }
};