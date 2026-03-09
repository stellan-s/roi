import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ROI Control Plane",
  description: "Run and inspect quant pipeline jobs",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
