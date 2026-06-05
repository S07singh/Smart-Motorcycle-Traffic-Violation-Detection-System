"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Bike,
  Users,
  ShieldCheck,
  ShieldX,
  ScanLine,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  Download,
} from "lucide-react";

import MetricCard from "@/components/MetricCard";
import ViolationCard from "@/components/ViolationCard";
import PlateDisplay from "@/components/PlateDisplay";
import DetectionChart from "@/components/DetectionChart";
import { getVideoDownloadUrl } from "@/lib/api";

interface ImageResult {
  annotated_image: string;
  detections: { class_name: string; confidence: number; bbox: number[] }[];
  violations: string[];
  violation_details: {
    violation_type: string;
    class_name: string;
    confidence: number;
    bbox: number[];
    persons_count?: number;
  }[];
  plate_results: {
    raw_text: string;
    cleaned_text: string;
    confidence: number;
  }[];
  summary: {
    motorcycle_count: number;
    person_count: number;
    helmet_count: number;
    no_helmet_count: number;
    license_plate_count: number;
    is_triple_riding: boolean;
    has_no_helmet: boolean;
  };
}

interface VideoResult {
  type: "video";
  job_id: string;
  status: string;
  progress: number;
  result: {
    total_frames: number;
    total_detections: number;
    max_persons_in_frame: number;
    total_no_helmets: number;
    violations: string[];
    unique_plates: { text: string; confidence: number }[];
  };
}

type ResultData = ImageResult | VideoResult;

function isVideoResult(data: ResultData): data is VideoResult {
  return (data as VideoResult).type === "video";
}

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;
  const [data, setData] = useState<ResultData | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem(id);
    if (stored) {
      setData(JSON.parse(stored));
    }
  }, [id]);

  if (!data) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-20 text-center">
        <p className="text-[#8892a8]">Loading results...</p>
      </div>
    );
  }

  // ── Video results ──
  if (isVideoResult(data)) {
    const r = data.result;
    const hasViolations = r.violations.length > 0;

    const chartData = [
      { name: "Frames", count: r.total_frames, color: "#06b6d4" },
      { name: "Detections", count: r.total_detections, color: "#8b5cf6" },
      { name: "Max Persons", count: r.max_persons_in_frame, color: "#f59e0b" },
      { name: "No Helmet", count: r.total_no_helmets, color: "#ef4444" },
    ];

    return (
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            id="back-home"
            onClick={() => router.push("/")}
            className="flex items-center gap-2 text-[#8892a8] hover:text-[#e8edf5] transition-colors text-sm"
          >
            <ArrowLeft className="w-4 h-4" />
            Analyze Another
          </button>
          <a
            href={getVideoDownloadUrl(data.job_id)}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-[#06b6d4] hover:text-[#0891b2] transition-colors"
          >
            <Download className="w-4 h-4" />
            Download Annotated Video
          </a>
        </div>

        {/* Status Banner */}
        <div
          className={`mb-8 p-5 rounded-xl border text-center font-semibold text-lg animate-slide-up ${
            hasViolations
              ? "bg-[#ef4444]/10 border-[#ef4444]/30 text-[#ef4444]"
              : "bg-[#22c55e]/10 border-[#22c55e]/30 text-[#22c55e]"
          }`}
        >
          <div className="flex items-center justify-center gap-3">
            {hasViolations ? (
              <AlertTriangle className="w-6 h-6" />
            ) : (
              <CheckCircle2 className="w-6 h-6" />
            )}
            {hasViolations
              ? `${r.violations.length} VIOLATION(S) DETECTED`
              : "ALL CLEAR — NO VIOLATIONS"}
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <MetricCard label="Total Frames" value={r.total_frames} icon={ScanLine} color="#06b6d4" delay={0} />
          <MetricCard label="Detections" value={r.total_detections} icon={ShieldCheck} color="#8b5cf6" delay={100} />
          <MetricCard label="Max Persons" value={r.max_persons_in_frame} icon={Users} color="#f59e0b" delay={200} />
          <MetricCard label="No Helmet" value={r.total_no_helmets} icon={ShieldX} color="#ef4444" delay={300} />
        </div>

        {/* Violations */}
        {r.violations.length > 0 && (
          <div className="mb-8">
            <h3 className="text-sm uppercase tracking-wider text-[#8892a8] font-medium mb-4">
              Violations Found
            </h3>
            <div className="space-y-3">
              {r.violations.map((v, i) => (
                <ViolationCard key={i} type={v} />
              ))}
            </div>
          </div>
        )}

        {/* Plates */}
        {r.unique_plates.length > 0 && (
          <div className="mb-8">
            <h3 className="text-sm uppercase tracking-wider text-[#8892a8] font-medium mb-4">
              License Plates Detected
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {r.unique_plates.map((p, i) => (
                <PlateDisplay
                  key={i}
                  plateText={p.text}
                  confidence={p.confidence}
                  index={i}
                />
              ))}
            </div>
          </div>
        )}

        {/* Chart */}
        <DetectionChart data={chartData} />
      </div>
    );
  }

  // ── Image results ──
  const imgData = data as ImageResult;
  const hasViolations = imgData.violations.length > 0;

  const chartData = [
    { name: "Motorcycles", count: imgData.summary.motorcycle_count, color: "#f59e0b" },
    { name: "Persons", count: imgData.summary.person_count, color: "#8b5cf6" },
    { name: "Helmets", count: imgData.summary.helmet_count, color: "#22c55e" },
    { name: "No Helmet", count: imgData.summary.no_helmet_count, color: "#ef4444" },
    { name: "Plates", count: imgData.summary.license_plate_count, color: "#06b6d4" },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-center justify-between mb-8 animate-slide-up">
        <button
          id="back-home"
          onClick={() => router.push("/")}
          className="flex items-center gap-2 text-[#8892a8] hover:text-[#e8edf5] transition-colors text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Analyze Another
        </button>
        <span className="text-xs text-[#4a5568] font-mono">
          ID: {id.slice(0, 12)}...
        </span>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        {/* LEFT — Annotated Image (3 cols) */}
        <div className="lg:col-span-3 animate-slide-up" style={{ animationDelay: "100ms" }}>
          <div className="glass-card overflow-hidden">
            <div className="p-3 border-b border-white/5 flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-[#ef4444]" />
              <div className="w-2.5 h-2.5 rounded-full bg-[#f59e0b]" />
              <div className="w-2.5 h-2.5 rounded-full bg-[#22c55e]" />
              <span className="ml-2 text-xs text-[#4a5568]">
                Annotated Detection Output
              </span>
            </div>
            <div className="p-2">
              <img
                src={`data:image/png;base64,${imgData.annotated_image}`}
                alt="Annotated detection result"
                className="w-full rounded-lg"
                id="annotated-image"
              />
            </div>
          </div>
        </div>

        {/* RIGHT — Results Panel (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Status Banner */}
          <div
            className={`p-5 rounded-xl border text-center font-semibold text-lg animate-slide-up ${
              hasViolations
                ? "bg-[#ef4444]/10 border-[#ef4444]/30 text-[#ef4444]"
                : "bg-[#22c55e]/10 border-[#22c55e]/30 text-[#22c55e]"
            }`}
            style={{ animationDelay: "150ms" }}
          >
            <div className="flex items-center justify-center gap-3">
              {hasViolations ? (
                <AlertTriangle className="w-6 h-6" />
              ) : (
                <CheckCircle2 className="w-6 h-6" />
              )}
              {hasViolations
                ? "VIOLATIONS DETECTED"
                : "ALL CLEAR"}
            </div>
          </div>

          {/* Metric Cards */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              label="Motorcycles"
              value={imgData.summary.motorcycle_count}
              icon={Bike}
              color="#f59e0b"
              delay={200}
            />
            <MetricCard
              label="Persons"
              value={imgData.summary.person_count}
              icon={Users}
              color="#8b5cf6"
              delay={250}
            />
            <MetricCard
              label="Helmets"
              value={imgData.summary.helmet_count}
              icon={ShieldCheck}
              color="#22c55e"
              delay={300}
            />
            <MetricCard
              label="No Helmet"
              value={imgData.summary.no_helmet_count}
              icon={ShieldX}
              color="#ef4444"
              delay={350}
            />
            <MetricCard
              label="Plates"
              value={imgData.summary.license_plate_count}
              icon={ScanLine}
              color="#06b6d4"
              delay={400}
            />
          </div>

          {/* Violation Cards */}
          {imgData.violation_details.length > 0 && (
            <div>
              <h3 className="text-sm uppercase tracking-wider text-[#8892a8] font-medium mb-3">
                Violations
              </h3>
              <div className="space-y-3">
                {imgData.violation_details.map((vd, i) => (
                  <ViolationCard
                    key={i}
                    type={vd.violation_type}
                    confidence={vd.confidence}
                    details={`Class: ${vd.class_name}`}
                    personsCount={vd.persons_count}
                  />
                ))}
              </div>
            </div>
          )}

          {/* License Plates */}
          {imgData.plate_results.length > 0 && (
            <div>
              <h3 className="text-sm uppercase tracking-wider text-[#8892a8] font-medium mb-3">
                License Plates
              </h3>
              <div className="space-y-4">
                {imgData.plate_results.map((pr, i) => (
                  <PlateDisplay
                    key={i}
                    plateText={pr.cleaned_text}
                    confidence={pr.confidence}
                    index={i}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Bar Chart */}
          <DetectionChart data={chartData} />

          {/* Analyze Another Button */}
          <button
            id="analyze-another"
            onClick={() => router.push("/")}
            className="btn-primary w-full text-base"
          >
            Analyze Another
          </button>
        </div>
      </div>
    </div>
  );
}
