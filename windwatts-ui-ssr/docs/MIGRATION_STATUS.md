# Handoff & Migration Status

This document serves as the primary handoff guide for the `windwatts-ui-ssr` project. It details the current status of the migration from the legacy `windwatts-ui` application, identifies known gaps, and provides an onboarding guide for new engineers.

**Last Updated:** 2025-12-29

## ✅ Completed Features

These features have been successfully ported or reimplemented in the SSR architecture:

- **App Shell**: Main layout structure, Header, and responsive grid system.
- **Map Integration**: Google Maps integration using Client Components (`src/components/Map.tsx`).
- **Data Fetching**: Server-side data fetching pipeline (`src/server/api.ts`) replacing client-side SWR.
- **Settings Management**: Global settings via `SettingsContext` and URL parameters.
- **Result Pane**:
  - `AnalysisResults`
  - `ProductionCard` & `ProductionDataTable`
  - `WindResourceCard` & `WindSpeedCard`
- **Search**: Location search bar functionality.

## 🚧 Gap Analysis: Missing Features

The following features exist in the legacy app (`windwatts-ui`) but are currently missing in the SSR version. These are categorized by priority.

### 1. Mobile Experience (High Priority)

- **Mobile Layouts**: The dedicated mobile views (`LayoutMobile.tsx`, `MobileBottomSheet.tsx`) are missing.
- **Current State**: The app is responsive (grids stack), but it lacks the polished drawer/sheet experience of the legacy app.
- **Missing Controls**: Mobile-specific map controls and search bar variations.

### 2. Actions & Tools (High Priority)

- **Export Data**: The `DownloadButton` and CSV export workflow is missing.
- **Sharing**: The UI component for `ShareButton` (generating short links or copy-paste URLs) is absent, though the URL structure itself supports deep linking.

### 3. Map Interactions (Medium Priority)

- **Out of Bounds Warnings**: Logic to warn users when panning outside supported regions (e.g., US only) is missing (`OutOfBoundsWarning.tsx`).

### 4. Branding & Polish (Low Priority)

- **Footer**: The NREL/WindWatts footer component is missing.
- **Loading States**: Granular loading feedback (skeletons/spinners) for specific map interactions needs refinement.

## 🚀 Onboarding Guide

### 1. The "URL is Truth" Principle

In `windwatts-ui-ssr`, if it's not in the URL, it's not the state (mostly). This is the biggest mental shift from the legacy app.

- **Legacy**: `SettingsContext` + `localStorage` was the source of truth.
- **SSR**: `URL Search Params` are the source of truth.
  - **Read**: Components read `searchParams` (Server) or `useSearchParams` (Client).
  - **Write**: Components update state by pushing a new URL (`router.push`).

**When adding a new setting:**

1.  Add default to `URL_PARAM_DEFAULTS` in `utils/urlParams.ts`.
2.  Update `SettingsProvider.tsx` to read/write this param.
3.  Update `page.tsx` to read this param and pass it to API calls.

### 2. Key Files Map

Where to find the new equivalent of the old files:

| Concept            | Legacy Path (`windwatts-ui`)        | SSR Path (`windwatts-ui-ssr`)        |
| :----------------- | :---------------------------------- | :----------------------------------- |
| **Main Page**      | `src/App.tsx` (or `Layout.tsx`)     | `app/page.tsx`                       |
| **Data Fetching**  | `src/hooks/useWindData.ts` (SWR)    | `src/server/api.ts` (Fetch)          |
| **Settings Logic** | `src/providers/SettingsContext.tsx` | `src/providers/SettingsProvider.tsx` |
| **API Client**     | `src/services/api.ts` (Client)      | `src/server/api.ts` (Server)         |

## ⚠️ Known Issues & Risks

### MUI Version Mismatch

- **Legacy**: Uses Material UI **v7** (latest).
- **SSR**: Uses Material UI **v5**.
- **Impact**: Copy-pasting components from `windwatts-ui` to `windwatts-ui-ssr` may fail if they use v6/v7 specific APIs. Check imports and props carefully.

### Prop Drilling

- **Issue**: Because we fetch data on the Server (`page.tsx`) but display it in Client Components (`RightPane.tsx`), we have to pass large data objects down as props.
- **Mitigation**: This is intentional to support SSR. Avoid tempted refactors to move fetching back to the client unless absolutely necessary for interaction latency.

## 📋 Prioritized Next Steps

1.  **Implement Mobile Parity**: Port `MobileBottomSheet` logic to work with the SSR layout.
2.  **Enable Data Export**: Re-implement the CSV download functionality using the server-fetched data.
3.  **UI Polish**: Add the missing footer and out-of-bounds warnings.
