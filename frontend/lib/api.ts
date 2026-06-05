/**
 * API client for the Motorcycle Violation Detection backend.
 *
 * All endpoints communicate with the FastAPI server specified by
 * NEXT_PUBLIC_API_URL (defaults to http://localhost:8000).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface DetectionResult {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: number[];
}

export interface ViolationDetail {
  violation_type: string;
  class_name: string;
  confidence: number;
  bbox: number[];
  persons_count?: number;
}

export interface PlateResult {
  raw_text: string;
  cleaned_text: string;
  confidence: number;
}

export interface SummaryStats {
  motorcycle_count: number;
  person_count: number;
  helmet_count: number;
  no_helmet_count: number;
  license_plate_count: number;
  is_triple_riding: boolean;
  has_no_helmet: boolean;
}

export interface ImageResponse {
  annotated_image: string; // base64 PNG
  detections: DetectionResult[];
  violations: string[];
  violation_details: ViolationDetail[];
  plate_results: PlateResult[];
  summary: SummaryStats;
}

export interface VideoJobResponse {
  job_id: string;
}

export interface VideoJobResult {
  total_frames: number;
  total_detections: number;
  max_persons_in_frame: number;
  total_no_helmets: number;
  violations: string[];
  unique_plates: { text: string; confidence: number }[];
}

export interface JobStatusResponse {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  result?: VideoJobResult | { error: string };
}

/**
 * Run image detection.
 */
export async function detectImage(
  file: File,
  confidence: number
): Promise<ImageResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("confidence", confidence.toString());
  const res = await fetch(`${API_BASE}/detect/image`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Submit a video for async processing.
 */
export async function submitVideo(
  file: File,
  confidence: number
): Promise<VideoJobResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("confidence", confidence.toString());
  const res = await fetch(`${API_BASE}/detect/video`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Poll the status of a video processing job.
 */
export async function pollJobStatus(
  jobId: string
): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/job/${jobId}/status`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Get the URL for downloading the annotated video.
 */
export function getVideoDownloadUrl(jobId: string): string {
  return `${API_BASE}/job/${jobId}/video`;
}
