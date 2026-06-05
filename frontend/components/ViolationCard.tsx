"use client";

import React from "react";
import { AlertTriangle } from "lucide-react";

interface ViolationCardProps {
  type: string;
  confidence?: number;
  details?: string;
  personsCount?: number;
}

export default function ViolationCard({
  type,
  confidence,
  details,
  personsCount,
}: ViolationCardProps) {
  return (
    <div
      id={`violation-card-${type.replace(/\s+/g, "-").toLowerCase()}`}
      className="violation-pulse relative overflow-hidden rounded-xl border border-[#ef4444]/30 bg-gradient-to-br from-[#ef4444]/10 to-[#ef4444]/5 p-5 transition-all duration-300 hover:border-[#ef4444]/50"
    >
      {/* Red accent line */}
      <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#ef4444]" />

      <div className="flex items-start gap-3 pl-2">
        <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-[#ef4444]/15 flex items-center justify-center">
          <AlertTriangle className="w-5 h-5 text-[#ef4444]" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-[#ef4444] text-base">{type}</h3>
          {details && (
            <p className="text-[#8892a8] text-sm mt-1">{details}</p>
          )}
          <div className="flex items-center gap-4 mt-2 text-xs text-[#4a5568]">
            {confidence !== undefined && (
              <span>
                Confidence:{" "}
                <span className="text-[#e8edf5] font-mono">
                  {(confidence * 100).toFixed(1)}%
                </span>
              </span>
            )}
            {personsCount !== undefined && (
              <span>
                Persons:{" "}
                <span className="text-[#e8edf5] font-mono">{personsCount}</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
