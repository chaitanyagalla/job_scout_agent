import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Job Scout",
  description: "Next.js frontend for the Job Scout FastAPI agent",
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
