"use client";

import React, { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Image as ImageIcon,
  Film,
  Zap,
  Shield,
  ScanLine,
  Bike,
} from "lucide-react";

import UploadZone from "@/components/UploadZone";
import ProgressPoller from "@/components/ProgressPoller";
import {
  detectImage,
  submitVideo,
  type ImageResponse,
  type JobStatusResponse,
} from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [mode, setMode] = useState<"image" | "video">("image");
  const [confidence, setConfidence] = useState(0.25);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoJobId, setVideoJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ── Image detection result: stored in sessionStorage, then navigate ──
  const handleImageResult = useCallback(
    (result: ImageResponse) => {
      const resultId = `img-${Date.now()}`;
      sessionStorage.setItem(resultId, JSON.stringify(result));
      router.push(`/results/${resultId}`);
    },
    [router]
  );

  // ── Video job completion ──
  const handleVideoComplete = useCallback(
    (status: JobStatusResponse) => {
      const resultId = status.job_id;
      sessionStorage.setItem(
        resultId,
        JSON.stringify({ type: "video", ...status })
      );
      router.push(`/results/${resultId}`);
    },
    [router]
  );

  // ── Run detection ──
  const handleRun = async () => {
    if (!selectedFile) return;
    setError(null);
    setIsProcessing(true);

    try {
      if (mode === "image") {
        const result = await detectImage(selectedFile, confidence);
        handleImageResult(result);
      } else {
        const { job_id } = await submitVideo(selectedFile, confidence);
        setVideoJobId(job_id);
      }
    } catch (err) {
      setError((err as Error).message);
      setIsProcessing(false);
    }
  };

  const imageAccept = ".jpg,.jpeg,.png,.bmp";
  const videoAccept = ".mp4,.avi,.mov,.mkv";

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      {/* ── Hero Section ── */}
      <div className="text-center mb-12 animate-slide-up">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#06b6d4]/10 border border-[#06b6d4]/20 text-[#06b6d4] text-xs font-medium mb-6">
          <Zap className="w-3.5 h-3.5" />
          Dual YOLOv8 + PaddleOCR Pipeline
        </div>
        <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
          <span className="bg-gradient-to-r from-[#06b6d4] via-[#0891b2] to-[#8b5cf6] bg-clip-text text-transparent">
            Traffic Violation
          </span>{" "}
          <br className="md:hidden" />
          Detection
        </h2>
        <p className="text-[#8892a8] text-lg max-w-xl mx-auto leading-relaxed">
          Upload an image or video to detect helmet violations, triple riding,
          and extract Indian license plate numbers using AI.
        </p>
      </div>

      {/* ── Feature Pills ── */}
      <div className="flex flex-wrap justify-center gap-3 mb-10 animate-slide-up" style={{ animationDelay: "100ms" }}>
        {[
          { icon: Shield, label: "Helmet Detection", color: "#22c55e" },
          { icon: Bike, label: "Triple Riding", color: "#ef4444" },
          { icon: ScanLine, label: "Plate OCR", color: "#06b6d4" },
        ].map((feat) => (
          <div
            key={feat.label}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-[#111a32] border border-white/5 text-sm"
          >
            <feat.icon className="w-4 h-4" style={{ color: feat.color }} />
            <span className="text-[#8892a8]">{feat.label}</span>
          </div>
        ))}
      </div>

      {/* ── Mode Toggle ── */}
      <div
        className="flex justify-center mb-8 animate-slide-up"
        style={{ animationDelay: "150ms" }}
      >
        <div className="inline-flex rounded-xl bg-[#111a32] border border-white/5 p-1">
          <button
            id="mode-image"
            onClick={() => {
              setMode("image");
              setSelectedFile(null);
              setVideoJobId(null);
              setError(null);
            }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
              mode === "image"
                ? "bg-[#06b6d4]/15 text-[#06b6d4] shadow-sm"
                : "text-[#8892a8] hover:text-[#e8edf5]"
            }`}
          >
            <ImageIcon className="w-4 h-4" />
            Image
          </button>
          <button
            id="mode-video"
            onClick={() => {
              setMode("video");
              setSelectedFile(null);
              setVideoJobId(null);
              setError(null);
            }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
              mode === "video"
                ? "bg-[#06b6d4]/15 text-[#06b6d4] shadow-sm"
                : "text-[#8892a8] hover:text-[#e8edf5]"
            }`}
          >
            <Film className="w-4 h-4" />
            Video
          </button>
        </div>
      </div>

      {/* ── Upload Zone ── */}
      <div className="animate-slide-up" style={{ animationDelay: "200ms" }}>
        <UploadZone
          onFileSelect={(f) => {
            setSelectedFile(f);
            setError(null);
          }}
          accept={mode === "image" ? imageAccept : videoAccept}
          mode={mode}
        />
      </div>

      {/* ── Confidence Slider ── */}
      <div
        className="mt-8 glass-card p-6 animate-slide-up"
        style={{ animationDelay: "250ms" }}
      >
        <div className="flex items-center justify-between mb-3">
          <label
            htmlFor="confidence-slider"
            className="text-sm text-[#8892a8] font-medium"
          >
            Detection Confidence Threshold
          </label>
          <span className="text-sm font-mono text-[#06b6d4] font-bold">
            {confidence.toFixed(2)}
          </span>
        </div>
        <input
          id="confidence-slider"
          type="range"
          min="0.10"
          max="0.95"
          step="0.05"
          value={confidence}
          onChange={(e) => setConfidence(parseFloat(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-[10px] text-[#4a5568] mt-1">
          <span>More detections</span>
          <span>More precise</span>
        </div>
      </div>

      {/* ── Error message ── */}
      {error && (
        <div className="mt-6 p-4 rounded-xl bg-[#ef4444]/10 border border-[#ef4444]/30 text-[#ef4444] text-sm animate-slide-up">
          {error}
        </div>
      )}

      {/* ── Run Button ── */}
      {!videoJobId && (
        <div className="mt-8 animate-slide-up" style={{ animationDelay: "300ms" }}>
          <button
            id="run-detection"
            onClick={handleRun}
            disabled={!selectedFile || isProcessing}
            className="btn-primary w-full text-lg flex items-center justify-center gap-3"
          >
            {isProcessing ? (
              <>
                <svg
                  className="animate-spin w-5 h-5"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="3"
                    className="opacity-25"
                  />
                  <path
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    fill="currentColor"
                    className="opacity-75"
                  />
                </svg>
                Processing...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Run Detection
              </>
            )}
          </button>
        </div>
      )}

      {/* ── Video Progress Poller ── */}
      {videoJobId && (
        <div className="mt-8">
          <ProgressPoller
            jobId={videoJobId}
            onComplete={handleVideoComplete}
            onError={(err) => {
              setError(err);
              setVideoJobId(null);
              setIsProcessing(false);
            }}
          />
        </div>
      )}
    </div>
  );
}
