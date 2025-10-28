"use client";

import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { UnitsContext, defaultUnitValues } from "../providers/UnitsContext";

export function UnitsProvider({ children }: { children: React.ReactNode }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const windspeed = (searchParams.get("windspeedUnit") as string) || defaultUnitValues.windspeed;
  const output = (searchParams.get("outputUnit") as string) || defaultUnitValues.output;

  const units = { windspeed, output };

  const setUnits = () => {};
  const updateUnit = (key: string, value: string) => {
    const params = new URLSearchParams(searchParams as any);
    if (key === "windspeed") params.set("windspeedUnit", value);
    if (key === "output") params.set("outputUnit", value);
    router.replace(`${pathname}?${params.toString()}`);
  };
  const updateUnits = (newValues: { windspeed?: string; output?: string }) => {
    const params = new URLSearchParams(searchParams as any);
    if (newValues.windspeed) params.set("windspeedUnit", newValues.windspeed);
    if (newValues.output) params.set("outputUnit", newValues.output);
    router.replace(`${pathname}?${params.toString()}`);
  };

  return (
    <UnitsContext.Provider value={{ units, setUnits, updateUnit, updateUnits }}>
      {children}
    </UnitsContext.Provider>
  );
}
