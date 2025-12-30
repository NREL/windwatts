import { useCallback, useEffect, useRef, useState } from "react";

export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void] {
  const isMountedRef = useRef(false);
  const [storedValue, setStoredValue] = useState<T>(initialValue);

  useEffect(() => {
    isMountedRef.current = true;
    try {
      const item = window.localStorage.getItem(key);
      if (item != null) {
        setStoredValue(JSON.parse(item));
      }
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
    }
  }, [key]);

  const setValue = useCallback(
    (value: T | ((val: T) => T)) => {
      setStoredValue((currentValue) => {
        const valueToStore =
          value instanceof Function ? value(currentValue) : value;
        try {
          if (isMountedRef.current) {
            window.localStorage.setItem(key, JSON.stringify(valueToStore));
          }
        } catch (error) {
          console.warn(`Error setting localStorage key "${key}":`, error);
        }
        return valueToStore;
      });
    },
    [key]
  );

  return [storedValue, setValue];
}
