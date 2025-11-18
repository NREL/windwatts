import { useState, useMemo, useContext } from "react";
import useSWR from "swr";
import { getCSVFile, getBatchCSVFiles, getNearestGridLocation } from "../services/api";
import { downloadWindDataCSV, downloadWindDataZIP } from "../services/download";
import { SettingsContext } from "../providers/SettingsContext";
import { GridLocation } from "../types";

export const useDownloadCSVFile = () => {
  const [isDownloading, setIsDownloading] = useState(false);
  const { currentPosition, preferredModel: dataModel } = useContext(SettingsContext);
  const { lat, lng } = currentPosition || {};

  const canDownload = !!(lat && lng && dataModel);

  const downloadFile = async (gridLat: number, gridLng: number, gridIndex: string) => {
    try {
      setIsDownloading(true);
      const response = await getCSVFile({
        gridIndex: gridIndex,
        dataModel: dataModel!,
      });
      await downloadWindDataCSV(response, gridLat, gridLng);
      return { success: true };
    } catch (error) {
      console.error("Download failed:", error);
      return { success: false, error };
    } finally {
      setIsDownloading(false);
    }
  };

  const downloadBatchFiles = async (gridLocations: GridLocation[]) => {
    try {
      setIsDownloading(true);
      const response = await getBatchCSVFiles({
        gridLocations: gridLocations,
        dataModel: dataModel!,
      });
      await downloadWindDataZIP(response, lat!, lng!);
      return { success: true };
    } catch (error) {
      console.error("Batch Download failed:", error);
      return { success: false, error };
    } finally {
      setIsDownloading(false);
    }
  };

  return {
    canDownload,
    isDownloading,
    downloadFile,
    downloadBatchFiles
  };
};

export const useNearestGridLocation = (n_neighbors: number = 1) => {
  const { currentPosition, preferredModel: dataModel } =
    useContext(SettingsContext);
  const { lat, lng } = currentPosition || {};

  const shouldFetch = lat && lng && dataModel;

  // Memoize the SWR key to prevent unnecessary re-renders
  const swrKey = useMemo(() => {
    if (!shouldFetch) return null;
    return JSON.stringify({ lat, lng, n_neighbors, dataModel, type: "nearest-grid" });
  }, [shouldFetch, lat, lng, n_neighbors, dataModel]);

  const { isLoading, data, error, mutate: retry } = useSWR(
    swrKey,
    () =>
      getNearestGridLocation({
        lat: lat!,
        lng: lng!,
        n_neighbors: n_neighbors,
        dataModel: dataModel!,
      }),
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      revalidateIfStale: false,
      dedupingInterval: 60000, // Cache for 1 minute
    }
  );

  const gridLocations = data?.locations?.map((location: GridLocation) => ({
    latitude: location.latitude,
    longitude: location.longitude,
    index: location.index
  })) || [];

  return {
    gridLocations,
    isLoading,
    error: error?.message || error || null,
    hasData: gridLocations.length > 0,
    retry,
  };
};
