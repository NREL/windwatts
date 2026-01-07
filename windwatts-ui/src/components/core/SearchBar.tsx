import React, {
  useState,
  useEffect,
  useRef,
  forwardRef,
  useImperativeHandle,
} from "react";
import {
  TextField,
  Box,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Paper,
} from "@mui/material";
import { LocationOn, Clear, Settings } from "@mui/icons-material";
import { useGoogleMaps } from "../../hooks";

interface SearchBarProps {
  onPlaceSelected?: (place: google.maps.places.PlaceResult) => void;
  useGoogleAutocomplete?: boolean;
  onSettingsClick?: () => void;
}

export interface SearchBarRef {
  clearInput: () => void;
}

export const SearchBar = forwardRef<SearchBarRef, SearchBarProps>(
  (
    { onPlaceSelected, useGoogleAutocomplete = false, onSettingsClick },
    ref
  ) => {
    const [inputValue, setInputValue] = useState("");
    const [predictions, setPredictions] = useState<
      google.maps.places.AutocompletePrediction[]
    >([]);
    const [showPredictions, setShowPredictions] = useState(false);

    const inputRef = useRef<HTMLInputElement>(null);
    const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(
      null
    );

    // Load Google Maps API
    const { isLoaded: isGoogleMapsReady } = useGoogleMaps();

    // Clear input state only (no side effects)
    const clearInput = () => {
      setInputValue("");
      setShowPredictions(false);
    };

    // Expose clearInput method to parent
    useImperativeHandle(ref, () => ({
      clearInput,
    }));

    // Initialize Google Autocomplete for native input
    useEffect(() => {
      if (
        useGoogleAutocomplete &&
        isGoogleMapsReady &&
        inputRef.current &&
        window.google
      ) {
        autocompleteRef.current = new window.google.maps.places.Autocomplete(
          inputRef.current,
          {
            types: ["geocode", "establishment"],
            fields: ["place_id", "geometry", "formatted_address", "name"],
          }
        );

        autocompleteRef.current.addListener("place_changed", () => {
          const place = autocompleteRef.current?.getPlace();
          if (place && place.geometry?.location && onPlaceSelected) {
            onPlaceSelected(place);
            setInputValue(place.formatted_address || "");
            setShowPredictions(false);
          }
        });

        return () => {
          if (autocompleteRef.current) {
            window.google.maps.event.clearInstanceListeners(
              autocompleteRef.current
            );
          }
        };
      }
    }, [useGoogleAutocomplete, isGoogleMapsReady, onPlaceSelected]);

    // Custom prediction search (when not using Google Autocomplete)
    useEffect(() => {
      if (
        !useGoogleAutocomplete &&
        isGoogleMapsReady &&
        inputValue.length >= 2
      ) {
        const service = new window.google.maps.places.AutocompleteService();

        service.getPlacePredictions(
          {
            input: inputValue,
            types: ["geocode", "establishment"],
          },
          (predictions, status) => {
            if (
              status === window.google.maps.places.PlacesServiceStatus.OK &&
              predictions
            ) {
              setPredictions(predictions);
              setShowPredictions(true);
            } else {
              setPredictions([]);
              setShowPredictions(false);
            }
          }
        );
      } else if (!useGoogleAutocomplete) {
        setPredictions([]);
        setShowPredictions(false);
      }
    }, [inputValue, isGoogleMapsReady, useGoogleAutocomplete]);

    const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      setInputValue(event.target.value);
    };

    const handlePredictionClick = async (
      prediction: google.maps.places.AutocompletePrediction
    ) => {
      if (!window.google) return;

      const service = new window.google.maps.places.PlacesService(
        document.createElement("div")
      );

      service.getDetails(
        {
          placeId: prediction.place_id,
          fields: ["place_id", "geometry", "formatted_address", "name"],
        },
        (place, status) => {
          if (
            status === window.google.maps.places.PlacesServiceStatus.OK &&
            place &&
            onPlaceSelected
          ) {
            onPlaceSelected(place);
            setInputValue(place.formatted_address || prediction.description);
            setShowPredictions(false);
          }
        }
      );
    };

    const handleClear = () => {
      clearInput();
      if (inputRef.current) {
        inputRef.current.focus();
      }
    };

    if (useGoogleAutocomplete) {
      return (
        <Box sx={{ position: "relative", width: "100%" }}>
          <input
            ref={inputRef}
            id="search-bar-input"
            type="text"
            placeholder="Enter a city, address, or landmark"
            value={inputValue}
            onChange={handleInputChange}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck="false"
            style={{
              width: "100%",
              padding: "12px 16px",
              paddingRight: onSettingsClick
                ? "88px"
                : inputValue
                  ? "48px"
                  : "16px",
              border: "1px solid #ddd",
              borderRadius: "8px",
              fontSize: "16px",
              outline: "none",
              backgroundColor: "white",
              boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
            }}
            onFocus={(e) => {
              e.target.style.borderColor = "#1976d2";
              e.target.style.boxShadow =
                "0 0 0 3px rgba(25, 118, 210, 0.2), 0 2px 8px rgba(0, 0, 0, 0.1)";
            }}
            onBlur={(e) => {
              e.target.style.borderColor = "#ddd";
              e.target.style.boxShadow = "0 2px 8px rgba(0, 0, 0, 0.1)";
            }}
          />
          <Box
            sx={{
              position: "absolute",
              right: "8px",
              top: "50%",
              transform: "translateY(-50%)",
              display: "flex",
              gap: 1,
            }}
          >
            {inputValue && (
              <IconButton
                size="small"
                onClick={handleClear}
                sx={{
                  color: "#666",
                  p: 1,
                  minWidth: "auto",
                  width: "32px",
                  height: "32px",
                  backgroundColor: "rgba(255, 255, 255, 0.9)",
                  "&:hover": {
                    backgroundColor: "rgba(255, 255, 255, 1)",
                    color: "#333",
                  },
                }}
              >
                <Clear sx={{ fontSize: 18 }} />
              </IconButton>
            )}
            {onSettingsClick && (
              <IconButton
                size="small"
                onClick={onSettingsClick}
                sx={{
                  color: "#666",
                  p: 1,
                  minWidth: "auto",
                  width: "32px",
                  height: "32px",
                  backgroundColor: "rgba(255, 255, 255, 0.9)",
                  "&:hover": {
                    backgroundColor: "rgba(255, 255, 255, 1)",
                    color: "#333",
                  },
                }}
              >
                <Settings sx={{ fontSize: 18 }} />
              </IconButton>
            )}
          </Box>
        </Box>
      );
    }

    return (
      <Box sx={{ position: "relative", width: "100%" }}>
        <TextField
          fullWidth
          placeholder="Enter a city, address, or landmark"
          value={inputValue}
          onChange={handleInputChange}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
          spellCheck="false"
          InputProps={{
            endAdornment: inputValue && (
              <IconButton size="small" onClick={handleClear}>
                <Clear />
              </IconButton>
            ),
          }}
        />
        {showPredictions && predictions.length > 0 && (
          <Paper
            sx={{
              position: "absolute",
              top: "100%",
              left: 0,
              right: 0,
              zIndex: 1000,
              maxHeight: 300,
              overflow: "auto",
              mt: 1,
            }}
          >
            <List>
              {predictions.map((prediction) => (
                <ListItem
                  key={prediction.place_id}
                  component="button"
                  onClick={() => handlePredictionClick(prediction)}
                  sx={{ py: 1, textAlign: "left", width: "100%" }}
                >
                  <ListItemIcon>
                    <LocationOn color="action" />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      prediction.structured_formatting?.main_text ||
                      prediction.description
                    }
                    secondary={prediction.structured_formatting?.secondary_text}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        )}
      </Box>
    );
  }
);

SearchBar.displayName = "SearchBar";
