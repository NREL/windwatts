# Frontend Development (UI)

The frontend is a React application built with Vite and Material-UI.

## Prerequisites

-   Node.js >= 22.14.0
-   Yarn

## Setup

```bash
cd windwatts-ui
yarn install
```

## Configuration

Create `.env.local` in `windwatts-ui/` (this overrides `.env.development`):

```shell
VITE_API_BASE_URL=http://windwatts-proxy:80
VITE_MAP_API_KEY=YOUR_GOOGLE_MAPS_API_KEY
VITE_MAP_ID=YOUR_MAP_ID
```

For production, create `.env.production`.

## Development

Start the dev server:

```bash
yarn dev
```

## Testing & Linting

-   **Lint**: `yarn lint`
-   **Test**: `yarn test`
-   **Format**: `yarn format`
