"use client";

import React from "react";
import type { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: number;
  icon: LucideIcon;
  color: string;   // Tailwind-compatible color, e.g. "#06b6d4"
  delay?: number;   // animation delay in ms
}

export default function MetricCard({
  label,
  value,
  icon: Icon,
  color,
  delay = 0,
}: MetricCardProps) {
  return (
    <div
      id={`metric-${label.replace(/\s+/g, "-").toLowerCase()}`}
      className="metric-card p-5 animate-slide-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${color}15` }}
        >
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
        <span className="text-xs uppercase tracking-wider text-[#8892a8] font-medium">
          {label}
        </span>
      </div>
      <div className="animate-count-up" style={{ animationDelay: `${delay + 200}ms` }}>
        <span
          className="text-3xl font-bold font-mono"
          style={{ color }}
        >
          {value}
        </span>
      </div>
    </div>
  );
}
