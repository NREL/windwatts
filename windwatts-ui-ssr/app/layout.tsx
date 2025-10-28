import Providers from "./providers";

export const metadata = {
  title: "Windwatts UI (SSR)",
  description: "Server-side rendered Windwatts UI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="emotion-insertion-point" content="" />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
