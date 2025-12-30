/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ["@mui/material", "@mui/icons-material", "@mui/system"]
  }
};

export default nextConfig;
