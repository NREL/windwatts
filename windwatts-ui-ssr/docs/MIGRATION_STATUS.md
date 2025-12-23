# Migration Status

This document tracks the feature parity between the legacy `windwatts-ui` (Vite/React) and the new `windwatts-ui-ssr` (Next.js/App Router) application.

**Last Updated:** 2025-12-16

## ✅ Completed Features

These features have been successfully ported or reimplemented in the SSR architecture:

-   **App Shell**: Main layout structure, Header, and responsive grid system.
-   **Map Integration**: Google Maps integration using Client Components (`src/components/Map.tsx`).
-   **Data Fetching**: Server-side data fetching pipeline (`src/server/api.ts`) replacing client-side SWR.
-   **Settings Management**: Global settings via `SettingsContext` and URL parameters.
-   **Result Pane**:
    -   `AnalysisResults`
    -   `ProductionCard` & `ProductionDataTable`
    -   `WindResourceCard` & `WindSpeedCard`
-   **Search**: Location search bar functionality.

## 🚧 Missing / To Be Migrated

The following features exist in the legacy app but are currently missing or incomplete in the SSR version:

### 1. Mobile Experience
-   **Mobile Layouts**: The dedicated mobile views (`LayoutMobile.tsx`, `MobileBottomSheet.tsx`) are not yet implemented. The current app is responsive but lacks the mobile-optimized drawer/sheet experience.
-   **Mobile Controls**: Mobile-specific map controls and search bar variations.

### 2. Actions & Tools
-   **Export Data**: The `DownloadButton` and CSV export workflow is missing.
-   **Sharing**: The `ShareButton` (generating short links or copy-paste URLs) is missing (though the URL structure supports it, the UI component is absent).

### 3. Map Interactions
-   **Out of Bounds Warnings**: Logic to warn users when panning outside supported regions (e.g., US only) is missing (`OutOfBoundsWarning.tsx`).

### 4. Branding & Polish
-   **Footer**: The NREL/WindWatts footer component is missing.
-   **Loading States**: Granular loading feedback (skeletons/spinners) for specific map interactions needs refinement.

## 📝 Notes for Contributors

-   **Priorities**: The "Actions & Tools" (Export) and "Mobile Experience" are the highest priority gaps to close.
-   **Reference**: Refer to the legacy codebase in `windwatts-ui/src/components` for the original implementation logic.


