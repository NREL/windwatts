# Architecture Guide

This document explains the architectural decisions and patterns used in the `windwatts-ui-ssr` application.

## Core Stack

-   **Framework**: Next.js 14 (App Router)
-   **Language**: TypeScript
-   **UI Library**: Material UI (MUI) v5
-   **Maps**: `@react-google-maps/api`

## Key Patterns

### 1. Launch in Context (URL as State)

One of the primary goals of this migration is to enable "Launch in Context". This means that the application's critical state (location, settings, selected models) is reflected in the URL search parameters.

-   **Legacy Approach**: State was persisted in `localStorage`. Sharing a link didn't guarantee the recipient saw the same view.
-   **SSR Approach**:
    -   State is read from **URL Search Params** (`?lat=...&lng=...&hubHeight=...`).
    -   When a user changes a setting, we update the URL (using `router.push` or `window.history`).
    -   `page.tsx` reads these params to fetch the correct data on the server.

### 2. Server-Side Rendering (SSR) & Data Fetching

We moved from Client-Side Rendering (CSR) to Server-Side Rendering to improve performance and data locality.

-   **Legacy (CSR)**: The page loaded empty, then `useSWR` hooks fetched data from the API. This often led to waterfalls.
-   **SSR (Next.js)**:
    -   **Entry Point**: `src/app/page.tsx` is an Async Server Component.
    -   **Fetching**: It calls `src/server/api.ts`, which fetches all necessary data (Wind Data, Production Data) in parallel on the server before rendering.
    -   **Result**: The initial HTML received by the browser contains the fully rendered data tables and cards.

### 3. Server vs. Client Components

We strictly separate Server and Client components to optimize bundle size and hydration.

| Component Type | Use Case | Examples |
| :--- | :--- | :--- |
| **Server Components** | Data fetching, Layout structure, Static content | `page.tsx`, `layout.tsx`, `AnalysisResults.tsx` |
| **Client Components** | Interactive elements, Browser APIs (Maps, Window), State (Context) | `Map.tsx`, `SettingsModal.tsx`, `SettingsProvider.tsx` |

**Note**: Client Components usually have the `"use client"` directive at the top of the file.

### 4. Map Integration

The Google Map is inherently a client-side interaction. We wrap it in a dedicated Client Component (`src/components/Map.tsx`) to isolate it from the Server Components.

-   **Synchronization**: The Map component listens to URL changes to update its center/zoom, and updates the URL when the user drags the map (debounced).

## Directory Structure

-   `src/app/`: Next.js App Router pages and layouts.
-   `src/components/`: React components (atomic designish).
-   `src/server/`: Server-side utilities and API fetchers.
-   `src/providers/`: React Context providers (Theme, Settings).
-   `src/utils/`: Shared helper functions.


