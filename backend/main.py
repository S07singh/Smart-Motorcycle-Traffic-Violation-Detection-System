"""
Smart Motorcycle Traffic Violation Detection API.

FastAPI application with:
  - Dual YOLOv8 model loading at startup (lifespan)
  - PaddleOCR initialisation at startup
  - Synchronous image detection endpoint
  - Async video processing with background tasks
  - In-memory job storage (Redis-backed in production via Celery)
"""

import base64
import os
import uuid
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from utils.detector import load_model, load_coco_model, run_detection, detect_motorcycles, detect_persons_coco
from utils.ocr_engine import init_ocr_reader, extract_plate_text
from utils.violation_checker import check_violations, check_triple_riding
from utils.visualizer import draw_detections

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]


class ViolationDetail(BaseModel):
    violation_type: str
    class_name: str
    confidence: float
    bbox: List[int]
    persons_count: Optional[int] = None


class PlateResult(BaseModel):
    raw_text: str
    cleaned_text: str
    confidence: float


class SummaryStats(BaseModel):
    motorcycle_count: int
    person_count: int
    helmet_count: int
    no_helmet_count: int
    license_plate_count: int
    is_triple_riding: bool
    has_no_helmet: bool


class ImageResponse(BaseModel):
    annotated_image: str  # base64 PNG
    detections: List[DetectionResult]
    violations: List[str]
    violation_details: List[ViolationDetail]
    plate_results: List[PlateResult]
    summary: SummaryStats


class VideoJobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending | processing | completed | failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

models: Dict[str, Any] = {}
jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Lifespan: load models ONCE at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load YOLOv8 models and PaddleOCR at startup, clear on shutdown."""
    model_dir = os.path.join(os.path.dirname(__file__), "model")

    custom_path = os.path.join(model_dir, "best.pt")
    coco_path = os.path.join(model_dir, "yolov8n.pt")

    if not os.path.exists(custom_path):
        raise RuntimeError(f"Custom model not found at {custom_path}")
    if not os.path.exists(coco_path):
        raise RuntimeError(f"COCO model not found at {coco_path}")

    models["custom"] = load_model(custom_path)
    models["coco"] = load_coco_model(coco_path)
    models["ocr"] = init_ocr_reader()

    yield

    models.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Motorcycle Violation Detection API",
    description="AI-powered traffic violation detection using dual YOLOv8 models and PaddleOCR.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_base64(image_bgr: np.ndarray) -> str:
    """Convert a BGR OpenCV image to a base64-encoded PNG string."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".png", image_rgb)
    return base64.b64encode(buffer).decode("utf-8")


def _process_single_image(
    image_bgr: np.ndarray,
    confidence: float,
) -> Dict[str, Any]:
    """Run full detection pipeline on a single image."""
    custom_model = models["custom"]
    coco_model = models["coco"]
    ocr_reader = models["ocr"]

    # Run custom model for helmet/no_helmet/license_plate
    detections = run_detection(custom_model, image_bgr, confidence=confidence)

    # Run COCO model for motorcycle + person detection
    motorcycles = detect_motorcycles(coco_model, image_bgr, confidence=confidence)
    coco_persons = detect_persons_coco(coco_model, image_bgr, confidence=confidence)

    # Per-motorcycle triple riding check using full-body COCO persons +
    # helmet/no_helmet detections as rider proxies for occluded riders
    triple_riding_results = check_triple_riding(motorcycles, coco_persons, detections)

    # Full violation report
    violation_report = check_violations(detections, triple_riding_results)

    # person_count = helmet detections + no_helmet detections
    # Each face-level detection from the custom model = exactly 1 rider.
    # This is more accurate than summing triple_riding_results (which can
    # double-count in dense traffic) or raw COCO person count (which includes
    # background pedestrians).
    violation_report["person_count"] = (
        violation_report["helmet_count"] + violation_report["no_helmet_count"]
    )

    # OCR on license plates — skip detections below 0.45 confidence to avoid
    # reading bike body markings that the custom model occasionally mistakes
    # for a license plate (false positives).
    plate_results = []
    for det in detections:
        if det["class_name"] == "license_plate" and det["confidence"] >= 0.45:
            ocr_result = extract_plate_text(ocr_reader, image_bgr, det["bbox"])
            plate_results.append({
                "raw_text": ocr_result["raw_text"],
                "cleaned_text": ocr_result["cleaned_text"],
                "confidence": ocr_result["confidence"],
            })

    # Draw annotations
    annotated = draw_detections(image_bgr, detections, violation_report, motorcycles)
    annotated_b64 = _image_to_base64(annotated)

    # Build summary
    summary = {
        "motorcycle_count": violation_report["motorcycle_count"],
        "person_count": violation_report["person_count"],
        "helmet_count": violation_report["helmet_count"],
        "no_helmet_count": violation_report["no_helmet_count"],
        "license_plate_count": violation_report["license_plate_count"],
        "is_triple_riding": violation_report["is_triple_riding"],
        "has_no_helmet": violation_report["has_no_helmet"],
    }

    # Serialise violation details (strip non-serialisable fields)
    violation_details = []
    for vd in violation_report["violation_details"]:
        violation_details.append({
            "violation_type": vd["violation_type"],
            "class_name": vd["class_name"],
            "confidence": vd["confidence"],
            "bbox": vd["bbox"],
            "persons_count": vd.get("persons_count"),
        })

    return {
        "annotated_image": annotated_b64,
        "detections": detections,
        "violations": violation_report["violations"],
        "violation_details": violation_details,
        "plate_results": plate_results,
        "summary": summary,
    }


def _process_video_background(
    job_id: str,
    video_path: str,
    confidence: float,
):
    """Background task: process video frame-by-frame, update job progress."""
    try:
        jobs[job_id]["status"] = "processing"

        custom_model = models["custom"]
        coco_model = models["coco"]
        ocr_reader = models["ocr"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["result"] = {"error": "Failed to open video file"}
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output annotated video
        out_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4", dir=tempfile.gettempdir()
        ).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        all_violations = set()
        all_plate_texts: List[Dict[str, Any]] = []
        total_detections = 0
        frame_count = 0
        max_persons_in_frame = 0
        total_no_helmets = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            detections = run_detection(custom_model, frame, confidence=confidence)
            motorcycles = detect_motorcycles(coco_model, frame, confidence=confidence)
            coco_persons = detect_persons_coco(coco_model, frame, confidence=confidence)
            triple_riding_results = check_triple_riding(motorcycles, coco_persons, detections)
            violation_report = check_violations(detections, triple_riding_results)
            # person_count uses effective_count (COCO + helmet proxy)
            violation_report["person_count"] = sum(
                tr["persons_count"] for tr in triple_riding_results
            ) if triple_riding_results else 0
            annotated_frame = draw_detections(frame, detections, violation_report, motorcycles)

            out.write(annotated_frame)

            total_detections += len(detections)
            for v in violation_report["violations"]:
                all_violations.add(v)
            max_persons_in_frame = max(
                max_persons_in_frame, violation_report["person_count"]
            )
            total_no_helmets += violation_report["no_helmet_count"]

            # OCR every 10th frame (same as original)
            if frame_count % 10 == 0:
                for det in detections:
                    if det["class_name"] == "license_plate":
                        ocr_result = extract_plate_text(ocr_reader, frame, det["bbox"])
                        if ocr_result["cleaned_text"]:
                            all_plate_texts.append({
                                "text": ocr_result["cleaned_text"],
                                "confidence": ocr_result["confidence"],
                            })

            # Update progress
            progress = frame_count / max(total_frames, 1)
            jobs[job_id]["progress"] = min(progress, 1.0)

        cap.release()
        out.release()

        # Deduplicate plate texts
        seen = set()
        unique_plates = []
        for pt in all_plate_texts:
            if pt["text"] not in seen:
                seen.add(pt["text"])
                unique_plates.append(pt)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["video_path"] = out_path
        jobs[job_id]["result"] = {
            "total_frames": frame_count,
            "total_detections": total_detections,
            "max_persons_in_frame": max_persons_in_frame,
            "total_no_helmets": total_no_helmets,
            "violations": list(all_violations),
            "unique_plates": unique_plates,
        }

        # Clean up input video
        try:
            os.unlink(video_path)
        except OSError:
            pass

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = {"error": str(e)}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=all(k in models for k in ("custom", "coco", "ocr")),
    )


@app.post("/detect/image", response_model=ImageResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    """
    Run detection on a single image.

    Accepts JPG/PNG via multipart form upload.
    Returns annotated image (base64), detections, violations, plate results, and summary.
    """
    if confidence < 0.1 or confidence > 0.95:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 0.95")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image. Please upload a valid JPG/PNG file.")

    result = _process_single_image(image_bgr, confidence)
    return ImageResponse(**result)


@app.post("/detect/video", response_model=VideoJobResponse)
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    """
    Submit a video for async detection processing.

    Returns a job_id to poll for status via /job/{job_id}/status.
    """
    if confidence < 0.1 or confidence > 0.95:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 0.95")

    # Determine file extension
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4, AVI, MOV, or MKV.")

    # Save uploaded video to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    contents = await file.read()
    tmp.write(contents)
    tmp.close()

    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "result": None,
        "video_path": None,
    }

    # Launch background processing
    background_tasks.add_task(_process_video_background, job_id, tmp.name, confidence)

    return VideoJobResponse(job_id=job_id)


@app.get("/job/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status and progress of a video processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"],
    )


@app.get("/job/{job_id}/video")
async def download_job_video(job_id: str):
    """Download the annotated video for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed")

    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Annotated video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"annotated_{job_id}.mp4",
    )
