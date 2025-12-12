import { useContext, useMemo } from "react";
import useSWR from "swr";
import { SettingsContext } from "../providers/SettingsContext";
import { getEnergyProduction } from "../services/api";
import { isOutOfBounds } from "../utils";
import { useLossAdjustedProductionData } from ".";

export const useProductionData = () => {
  const {
    currentPosition,
    hubHeight,
    powerCurve,
    preferredModel,
    lossAssumptionFactor,
  } = useContext(SettingsContext);
  // Always use ERA5 for production calculations, even when ensemble is selected
  const dataModel = preferredModel === "ensemble" ? "era5" : preferredModel;
  const { lat, lng } = currentPosition || {};
  const outOfBounds =
    lat && lng && dataModel ? isOutOfBounds(lat, lng, dataModel) : false;
  const shouldFetch =
    lat && lng && hubHeight && powerCurve && dataModel && !outOfBounds;

  // Memoize the SWR key to prevent unnecessary re-renders
  const swrKey = useMemo(() => {
    if (!shouldFetch) return null;
    return JSON.stringify({
      lat,
      lng,
      hubHeight,
      powerCurve,
      dataModel,
      period: "full",
    });
  }, [shouldFetch, lat, lng, hubHeight, powerCurve, dataModel]);

  const { isLoading, data, error } = useSWR(
    swrKey,
    () =>
      getEnergyProduction({
        lat: lat!,
        lng: lng!,
        hubHeight,
        powerCurve,
        dataModel,
        period: "full",
      }),
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      revalidateIfStale: false,
      dedupingInterval: 0,
    }
  );

  const productionDataWithLoss = useLossAdjustedProductionData(
    data,
    lossAssumptionFactor
  );

  return {
    productionData: productionDataWithLoss ?? data,
    isLoading,
    error,
    hasData: !!data,
    outOfBounds,
    dataModel,
    lat,
    lng,
  };
};
