"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface DetectionChartProps {
  data: {
    name: string;
    count: number;
    color: string;
  }[];
}

/* eslint-disable @typescript-eslint/no-explicit-any */
const CustomTooltip = ({
  active,
  payload,
}: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card px-4 py-2.5 text-sm">
        <p className="text-[#e8edf5] font-medium">{payload[0].payload.name}</p>
        <p className="text-[#8892a8]">
          Count:{" "}
          <span className="font-mono text-[#06b6d4] font-bold">
            {payload[0].value}
          </span>
        </p>
      </div>
    );
  }
  return null;
};

export default function DetectionChart({ data }: DetectionChartProps) {
  if (!data || data.length === 0) return null;

  return (
    <div
      id="detection-chart"
      className="glass-card p-6 animate-slide-up"
      style={{ animationDelay: "400ms" }}
    >
      <h3 className="text-sm uppercase tracking-wider text-[#8892a8] font-medium mb-4">
        Detection Class Distribution
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          data={data}
          margin={{ top: 5, right: 10, left: -10, bottom: 5 }}
          barCategoryGap="25%"
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />
          <XAxis
            dataKey="name"
            tick={{ fill: "#8892a8", fontSize: 12 }}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#8892a8", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            allowDecimals={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
          <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={50}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
