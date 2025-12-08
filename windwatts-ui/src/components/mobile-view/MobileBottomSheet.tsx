import React, {
  useState,
  useRef,
  forwardRef,
  useImperativeHandle,
  useEffect,
} from "react";
import { Box, IconButton } from "@mui/material";
import { Close } from "@mui/icons-material";
import { RightPane } from "../resultPane/RightPane";
import { SearchResultsList, MobileSearchBar } from "./components";
import { Footer } from "../Footer";
import {
  MobileBottomSheetProps,
  MobileBottomSheetRef,
  MobileSearchBarRef,
} from "./types";
import {
  MOBILE_BOTTOM_SHEET_HEIGHT,
  MOBILE_BOTTOM_SHEET_COLLAPSED_HEIGHT,
  FLAG_RESET_DELAY,
} from "../../constants";

export const MobileBottomSheet = forwardRef<
  MobileBottomSheetRef,
  MobileBottomSheetProps
>(({ isLoaded, onPlaceSelected }, ref) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [hasSelectedPlace, setHasSelectedPlace] = useState(false);
  const [searchPredictions, setSearchPredictions] = useState<
    google.maps.places.AutocompletePrediction[]
  >([]);
  const [isSearching, setIsSearching] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [hasInitialLocation, setHasInitialLocation] = useState(false);
  const searchBarRef = useRef<MobileSearchBarRef>(null);
  const isSettingFromSelectionRef = useRef(false);

  // Set initial location flag when component mounts (like desktop behavior)
  useEffect(() => {
    setHasInitialLocation(true);
  }, []);

  // Expose clearSearchInput method to parent
  useImperativeHandle(ref, () => ({
    clearSearchInput: () => {
      setInputValue("");
      setSearchPredictions([]);
      setIsSearching(false);
      setHasInitialLocation(true);
    },
    expandDrawer: () => {
      setIsExpanded(true);
    },
  }));

  const handlePlaceSelected = (place: google.maps.places.PlaceResult) => {
    onPlaceSelected(place);
    setHasSelectedPlace(true);
    setHasInitialLocation(true);
    setIsSearching(false);
    setSearchPredictions([]);
    // Set flag to prevent prediction search when updating input value
    isSettingFromSelectionRef.current = true;
    // Update input value to show the selected place (consistent with desktop)
    const newValue = place.formatted_address || place.name || "";
    setInputValue(newValue);
    // Reset flag after a short delay to allow the input update to complete
    setTimeout(() => {
      isSettingFromSelectionRef.current = false;
    }, FLAG_RESET_DELAY);
  };

  const handleSearchPredictions = (
    predictions: google.maps.places.AutocompletePrediction[],
    searching: boolean
  ) => {
    setSearchPredictions(predictions);
    setIsSearching(searching);
  };

  const handleInputChange = (value: string) => {
    setInputValue(value);
  };

  const handleClear = () => {
    setInputValue("");
    setSearchPredictions([]);
    setIsSearching(false);
    setHasSelectedPlace(false);
    // Show results when clearing (like desktop behavior)
    setHasInitialLocation(true);

    // Focus the search input after clearing
    setTimeout(() => {
      const searchInput = document.querySelector(
        "#mobile-search-input"
      ) as HTMLInputElement;
      if (searchInput) {
        searchInput.focus();
      }
    }, 100);
  };

  const handlePredictionClick = (
    prediction: google.maps.places.AutocompletePrediction
  ) => {
    // Create a dummy map element for PlacesService if needed
    const mapDiv = document.createElement("div");
    const map = new window.google.maps.Map(mapDiv);
    const placesService = new window.google.maps.places.PlacesService(map);

    const request = {
      placeId: prediction.place_id,
      fields: ["geometry", "name", "formatted_address", "place_id"],
    };

    placesService.getDetails(request, (place, status) => {
      if (
        status === window.google.maps.places.PlacesServiceStatus.OK &&
        place
      ) {
        handlePlaceSelected(place);
      }
    });
  };

  const expandDrawer = () => {
    setIsExpanded(true);
    // Focus the search input after animation
    setTimeout(() => {
      const searchInput = document.querySelector(
        "#mobile-search-input"
      ) as HTMLInputElement;
      if (searchInput) {
        searchInput.focus();
      }
    }, 350);
  };

  const collapseDrawer = () => {
    setIsExpanded(false);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    const startY = e.touches[0].clientY;

    const handleTouchMove = (e: TouchEvent) => {
      const currentY = e.touches[0].clientY;
      const deltaY = startY - currentY;

      // Only handle swipe up when collapsed, or swipe down when expanded
      if (!isExpanded && deltaY > 50) {
        if (e.cancelable) {
          e.preventDefault();
          e.stopPropagation();
        }
        expandDrawer();
      } else if (isExpanded && deltaY < -50) {
        if (e.cancelable) {
          e.preventDefault();
          e.stopPropagation();
        }
        collapseDrawer();
      }
    };

    const handleTouchEnd = () => {
      document.removeEventListener("touchmove", handleTouchMove);
      document.removeEventListener("touchend", handleTouchEnd);
    };

    document.addEventListener("touchmove", handleTouchMove, { passive: false });
    document.addEventListener("touchend", handleTouchEnd, { passive: true });
  };

  // Handle swipe down when expanded - this should work on the entire drawer
  const handleDrawerTouchStart = (e: React.TouchEvent) => {
    if (!isExpanded) return; // Only handle when expanded

    const startY = e.touches[0].clientY;
    const isScrollableArea = (e.target as Element).closest(
      '[data-scrollable="true"]'
    );

    // If touching a scrollable area, let it handle scrolling
    if (isScrollableArea) return;

    const handleTouchMove = (e: TouchEvent) => {
      const currentY = e.touches[0].clientY;
      const deltaY = startY - currentY;

      // Only handle significant downward swipes
      if (deltaY < -50) {
        if (e.cancelable) {
          e.preventDefault();
          e.stopPropagation();
        }
        collapseDrawer();
      }
    };

    const handleTouchEnd = () => {
      document.removeEventListener("touchmove", handleTouchMove);
      document.removeEventListener("touchend", handleTouchEnd);
    };

    document.addEventListener("touchmove", handleTouchMove, { passive: false });
    document.addEventListener("touchend", handleTouchEnd, { passive: true });
  };

  return (
    <Box
      sx={{
        position: "fixed",
        bottom: 0,
        left: 0,
        right: 0,
        height: isExpanded
          ? MOBILE_BOTTOM_SHEET_HEIGHT
          : MOBILE_BOTTOM_SHEET_COLLAPSED_HEIGHT, // Always show search bar
        bgcolor: "white",
        borderTopLeftRadius: 16,
        borderTopRightRadius: 16,
        boxShadow: "0 -2px 10px rgba(0,0,0,0.1)",
        transition: "height 0.3s ease-in-out",
        zIndex: 1300,
        display: "flex",
        flexDirection: "column",
        overscrollBehavior: "contain",
        touchAction: isExpanded ? "pan-y" : "auto",
      }}
      onTouchStart={handleDrawerTouchStart}
    >
      {/* Header with search bar */}
      <Box
        sx={{
          p: 2,
          borderBottom: isExpanded ? "1px solid #eee" : "none",
          cursor: "pointer",
          flexShrink: 0,
        }}
        onClick={() => {
          if (!isExpanded) {
            expandDrawer();
          }
        }}
        onTouchStart={handleTouchStart}
      >
        {/* Pull indicator */}
        <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
          <Box
            sx={{ width: 36, height: 4, bgcolor: "#ccc", borderRadius: 2 }}
          />
        </Box>

        {/* Search bar with close button */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          {isLoaded && (
            <Box sx={{ flex: 1 }}>
              <MobileSearchBar
                ref={searchBarRef}
                onSearchPredictions={handleSearchPredictions}
                onInputChange={handleInputChange}
                inputValue={inputValue}
                isSettingFromSelectionRef={isSettingFromSelectionRef}
              />
            </Box>
          )}
          {isExpanded && (
            <IconButton
              onClick={(e) => {
                e.stopPropagation();
                if (inputValue || hasSelectedPlace) {
                  // First click: clear the search and focus input
                  handleClear();
                } else {
                  // Second click: collapse the drawer
                  collapseDrawer();
                }
              }}
              sx={{
                bgcolor: "#f5f5f5",
                border: "1px solid #e5e7eb",
                "&:hover": {
                  bgcolor: "#e5e7eb",
                },
                width: 44,
                height: 44,
              }}
            >
              <Close sx={{ fontSize: 20, color: "#666" }} />
            </IconButton>
          )}
        </Box>
      </Box>

      {/* Content area */}
      <Box
        data-scrollable="true"
        sx={{
          flex: 1,
          overflow: "auto",
          display: isExpanded ? "block" : "none",
        }}
      >
        {isSearching && searchPredictions.length > 0 ? (
          <SearchResultsList
            predictions={searchPredictions}
            onPredictionClick={handlePredictionClick}
          />
        ) : hasSelectedPlace || hasInitialLocation ? (
          <>
            <RightPane />
            <Footer />
          </>
        ) : null}
      </Box>
    </Box>
  );
});

MobileBottomSheet.displayName = "MobileBottomSheet";
