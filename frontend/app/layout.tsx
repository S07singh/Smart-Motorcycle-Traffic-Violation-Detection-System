import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Smart Motorcycle Violation Detector — AI Traffic Surveillance",
  description:
    "AI-powered motorcycle traffic violation detection system using dual YOLOv8 models. Detects helmet violations, triple riding, and reads Indian license plates in real-time.",
  keywords: [
    "motorcycle violation detection",
    "helmet detection",
    "triple riding",
    "license plate recognition",
    "YOLOv8",
    "traffic surveillance",
    "AI",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#0a0f1e] text-[#e8edf5]">
        {/* Ambient grid pattern overlay */}
        <div className="fixed inset-0 grid-pattern pointer-events-none z-0" />

        {/* Navigation bar */}
        <nav className="relative z-10 border-b border-white/5 bg-[#0a0f1e]/80 backdrop-blur-md">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-cyan-700 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-cyan-500/20">
                MV
              </div>
              <div>
                <h1 className="text-lg font-semibold tracking-tight">
                  Motorcycle Violation Detector
                </h1>
                <p className="text-xs text-[#8892a8]">
                  AI Traffic Surveillance System
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-[#8892a8]">
              <span className="status-dot online" />
              <span>System Online</span>
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="relative z-10 flex-1">{children}</main>

        {/* Footer */}
        <footer className="relative z-10 border-t border-white/5 py-4 text-center text-xs text-[#4a5568]">
          Powered by YOLOv8 &amp; PaddleOCR — Smart Traffic Enforcement
        </footer>
      </body>
    </html>
  );
}
