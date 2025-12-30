# WindWatts UI (SSR)

This is the Next.js (App Router) migration of the WindWatts UI. It replaces the legacy client-side React application with a Server-Side Rendered architecture to improve performance, SEO, and "launch in context" capabilities.

## Quickstart

1.  **Install dependencies:**
    ```bash
    npm install
    ```

2.  **Set up environment variables:**
    Create a `.env.local` file in the root of this directory and add your API keys:
    ```bash
    NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_key_here
    NEXT_PUBLIC_API_URL=http://localhost:8000
    ```

3.  **Run the development server:**
    ```bash
    npm run dev
    ```

    Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Documentation

-   [**Architecture Guide**](docs/ARCHITECTURE.md): Learn about the shift to Server Components and URL-based state management.
-   [**Migration Status**](docs/MIGRATION_STATUS.md): Track feature parity with the legacy app and see what's left to build.

## Key Features

-   **Server-Side Rendering**: Initial data fetching happens on the server for faster First Contentful Paint.
-   **URL-Driven State**: Application state (location, settings) is synced to the URL, making every view shareable.
-   **Modern Stack**: Built with Next.js 14, React 18, and Material UI v5.


