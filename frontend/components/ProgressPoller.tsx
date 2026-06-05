"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { Loader2 } from "lucide-react";
import { pollJobStatus, type JobStatusResponse } from "@/lib/api";

interface ProgressPollerProps {
  jobId: string;
  onComplete: (result: JobStatusResponse) => void;
  onError?: (error: string) => void;
}

export default function ProgressPoller({
  jobId,
  onComplete,
  onError,
}: ProgressPollerProps) {
  const [status, setStatus] = useState<
    "pending" | "processing" | "completed" | "failed"
  >("pending");
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await pollJobStatus(jobId);
      setStatus(res.status);
      setProgress(res.progress);

      if (res.status === "completed") {
        if (intervalRef.current) clearInterval(intervalRef.current);
        onComplete(res);
      } else if (res.status === "failed") {
        if (intervalRef.current) clearInterval(intervalRef.current);
        const errResult = res.result as { error?: string } | undefined;
        onError?.(errResult?.error || "Processing failed");
      }
    } catch (err) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      onError?.((err as Error).message);
    }
  }, [jobId, onComplete, onError]);

  useEffect(() => {
    poll(); // initial poll
    intervalRef.current = setInterval(poll, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [poll]);

  const percentage = Math.round(progress * 100);

  return (
    <div id="progress-poller" className="glass-card p-8 text-center animate-slide-up">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Loader2 className="w-6 h-6 text-[#06b6d4] animate-spin" />
        <h3 className="text-lg font-semibold text-[#e8edf5]">
          {status === "pending"
            ? "Queuing video..."
            : status === "processing"
            ? "Processing video..."
            : status === "failed"
            ? "Processing failed"
            : "Complete!"}
        </h3>
      </div>

      {/* Progress bar */}
      <div className="relative w-full h-2 bg-[#111a32] rounded-full overflow-hidden mb-3">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500 ease-out"
          style={{
            width: `${percentage}%`,
            background:
              status === "failed"
                ? "#ef4444"
                : "linear-gradient(90deg, #06b6d4 0%, #0891b2 100%)",
            boxShadow:
              status === "failed"
                ? "0 0 12px rgba(239,68,68,0.4)"
                : "0 0 12px rgba(6,182,212,0.4)",
          }}
        />
      </div>

      <div className="flex items-center justify-between text-xs text-[#8892a8]">
        <span>
          Status:{" "}
          <span className="font-mono text-[#e8edf5] capitalize">{status}</span>
        </span>
        <span className="font-mono text-[#06b6d4]">{percentage}%</span>
      </div>
    </div>
  );
}
