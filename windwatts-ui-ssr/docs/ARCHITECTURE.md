# Architecture Guide

This document explains the architectural decisions, patterns, and development workflow for the `windwatts-ui-ssr` application.

## Core Stack

-   **Framework**: Next.js 14 (App Router)
-   **Language**: TypeScript
-   **UI Library**: Material UI (MUI) v5
-   **Maps**: `@react-google-maps/api`

## Development Workflow

This application can be run in two ways: as part of the full Docker ecosystem or as a standalone local application.

### 1. Run in Docker (Recommended)

This method ensures you are running with the exact same configuration as other services (API, Database).

```bash
# From the project root (windwatts/)
docker compose up windwatts-ui-ssr
```

-   **Access**: http://localhost:5174
-   **Hot Reload**: Enabled via volume mounts.
-   **Environment**: Uses `.env.development` provided in `windwatts-ui-ssr/`.
-   **Networking**: Connects to `windwatts-api` via the internal Docker network (through `nginx` proxy).

### 2. Run Locally (Standalone)

Useful for faster iteration on UI components without spinning up the backend services (though you will still need the API running for data).

```bash
# From the project root
cd windwatts-ui-ssr

# Install dependencies
npm install

# Start the dev server
npm run dev
```

-   **Access**: http://localhost:3000 (default Next.js port)
-   **Environment Variables**: ensure you have a `.env.local` file if you need to override defaults.
    -   `WINDWATTS_API_BASE`: If running the API locally, point this to `http://localhost:8000/api`.

## System Context

The `windwatts-ui-ssr` app operates within a larger ecosystem.

```mermaid
graph TD
    Client[Browser] -->|Requests Page| NextJS[Next.js Server (SSR)]
    Client -->|Interacts with Map| GoogleMaps[Google Maps API]
    
    subgraph Docker Network
        NextJS -->|Server-Side Data Fetch| Nginx[Nginx Proxy]
        Nginx -->|Proxy Request| API[WindWatts API (FastAPI)]
        API -->|Query| DB[(PostgreSQL)]
    end
```

### Key Components

1.  **Next.js Server**: Renders the initial HTML. It acts as an API client, fetching data from the backend **before** sending the page to the user.
2.  **WindWatts API**: The backend service providing wind data and production estimates.
3.  **Nginx Proxy**: In the Docker environment, Nginx sits in front of the API to handle routing and prevent CORS issues during server-side fetching if domains differed (though in Docker they are distinct services).
4.  **Google Maps**: Loaded purely on the client-side. The Next.js server does not render the map tiles; it only provides the container and configuration.

## Key Patterns

### 1. Launch in Context (URL as State)

One of the primary goals of this migration is to enable "Launch in Context". This means that the application's critical state (location, settings, selected models) is reflected in the URL search parameters.

-   **Legacy Approach**: State was persisted in `localStorage`. Sharing a link didn't guarantee the recipient saw the same view.
-   **SSR Approach**:
    -   **Source of Truth**: State is read from **URL Search Params** (`?lat=...&lng=...&hubHeight=...`).
    -   **Updates**: When a user changes a setting, we update the URL (using `router.push` or `window.history`).
    -   **Hydration**: `page.tsx` reads these params to fetch the correct data on the server.

### 2. Server-Side Rendering (SSR) & Data Fetching

We moved from Client-Side Rendering (CSR) to Server-Side Rendering to improve performance and data locality.

-   **Legacy (CSR)**: The page loaded empty, then `useSWR` hooks fetched data from the API. This often led to waterfalls.
-   **SSR (Next.js)**:
    -   **Entry Point**: `src/app/page.tsx` is an Async Server Component.
    -   **Fetching**: It calls `src/server/api.ts`, which fetches all necessary data (Wind Data, Production Data) in parallel on the server before rendering.
    -   **Result**: The initial HTML received by the browser contains the fully rendered data tables and cards.

### 3. Server vs. Client Components

We strictly separate Server and Client components to optimize bundle size and hydration.

#### Decision Tree

1.  **Does it need to fetch initial data?** -> **Server Component** (`page.tsx`)
2.  **Does it need access to headers/cookies?** -> **Server Component**
3.  **Does it have interactivity (onClick, onChange)?** -> **Client Component**
4.  **Does it use React Hooks (useState, useEffect)?** -> **Client Component**
5.  **Does it use Browser APIs (window, document, geolocation)?** -> **Client Component**

#### Data Flow

The typical data flow follows a strict hierarchy:

```
Page (Server Component)
  ├── 1. Reads URL Search Params
  ├── 2. Fetches Data (API)
  └── 3. Renders Layout
      │
      └──> Passes Data as Props -> RightPane (Client Component)
                                      ├── Renders Tabs
                                      ├── Renders Charts
                                      └── Uses State for UI toggles
```

This "Prop Drilling" is intentional to keep the data fetching logic on the server while allowing the leaf components to be interactive.

| Component Type | Use Case | Examples |
| :--- | :--- | :--- |
| **Server Components** | Data fetching, Layout structure, Static content | `page.tsx`, `layout.tsx`, `AnalysisResults.tsx` (container) |
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
