import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "upload.wikimedia.org" },
      { protocol: "https", hostname: "*.inaturalist.org" },
      { protocol: "https", hostname: "inaturalist-open-data.s3.amazonaws.com" },
      { protocol: "https", hostname: "static.inaturalist.org" },
    ],
  },
};

export default nextConfig;
