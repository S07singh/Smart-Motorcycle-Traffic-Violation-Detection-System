"use client";

import React from "react";
import { RectangleEllipsis } from "lucide-react";

interface PlateDisplayProps {
  plateText: string;
  confidence: number;
  index?: number;
}

export default function PlateDisplay({
  plateText,
  confidence,
  index = 0,
}: PlateDisplayProps) {
  return (
    <div
      id={`plate-display-${index}`}
      className="animate-slide-up"
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="plate-display px-6 py-4 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <RectangleEllipsis className="w-4 h-4 text-[#06b6d4]/60" />
          <span className="text-[10px] uppercase tracking-[0.3em] text-[#06b6d4]/60 font-semibold">
            License Plate
          </span>
        </div>
        <p className="text-2xl font-bold text-[#e8edf5] tracking-[0.2em]">
          {plateText || "—"}
        </p>
        <div className="mt-2 text-xs text-[#8892a8]">
          OCR Confidence:{" "}
          <span className="font-mono text-[#06b6d4]">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
