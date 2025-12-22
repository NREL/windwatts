import { useContext, useEffect } from "react";
import { SettingsContext } from "../providers/SettingsContext";
import { UnitsContext } from "../providers/UnitsContext";
import { useProductionData } from "./useProductionData";
import { useEnsembleTilesData } from "./useEnsembleTilesData";
import { KEY_AVERAGE_YEAR, KEY_KWH_PRODUCED } from "../constants";

export const useOutputUnit = () => {
  const { preferredModel } = useContext(SettingsContext);
  const { updateUnit } = useContext(UnitsContext);
  const { productionData: prodData } = useProductionData();
  const { productionData: ensembleData } = useEnsembleTilesData();

  useEffect(() => {
    const prodAvg = Number(
      prodData?.summary_avg_energy_production?.[KEY_AVERAGE_YEAR]?.[
        KEY_KWH_PRODUCED
      ] || 0
    );

    let shouldUseMWh = false;

    if (preferredModel === "ensemble") {
      // ensemble mode - auto-convert if either output >= 10,000
      const ensembleProduction = Number(ensembleData?.energy_production || 0);
      shouldUseMWh = ensembleProduction >= 10000 || prodAvg >= 10000;
    } else {
      // non-ensemble mode - auto-convert if average output >= 10,000
      shouldUseMWh = prodAvg >= 10000;
    }

    updateUnit("output", shouldUseMWh ? "MWh" : "kWh");
  }, [preferredModel, prodData, ensembleData, updateUnit]);
};
