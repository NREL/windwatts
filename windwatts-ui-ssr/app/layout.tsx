import Providers from "./providers";
import Header from "../src/components/Header";
import SettingsModal from "../src/components/SettingsModal";

export const metadata = {
  title: {
    default: "WindWatts UI (SSR)",
    template: "%s | WindWatts"
  },
  description: "Server-side rendered WindWatts UI",
  icons: {
    icon: "/icon.svg"
  },
  manifest: "/manifest.webmanifest",
  openGraph: {
    title: "WindWatts",
    description: "Wind resource and production estimates",
    url: "https://windwatts.nrel.gov",
    siteName: "WindWatts",
    locale: "en_US",
    type: "website"
  },
  twitter: {
    card: "summary_large_image",
    site: "@NREL",
    title: "WindWatts",
    description: "Wind resource and production estimates"
  }
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="emotion-insertion-point" content="" />
      </head>
      <body>
        <Providers>
          <Header />
          <SettingsModal />
          {children}
        </Providers>
      </body>
    </html>
  );
}
