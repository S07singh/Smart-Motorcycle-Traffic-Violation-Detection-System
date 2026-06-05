"""
Celery worker for production video processing.

In production, video jobs are dispatched to this Celery worker via Redis,
instead of using FastAPI BackgroundTasks. This allows horizontal scaling
of video processing across multiple worker pods.

Usage:
    celery -A worker worker --loglevel=info
"""

import os
import tempfile
from typing import List, Dict, Any

import cv2
from celery import Celery

from utils.detector import load_model, load_coco_model, run_detection, detect_motorcycles, detect_persons
from utils.ocr_engine import init_ocr_reader, extract_plate_text
from utils.violation_checker import check_violations, check_triple_riding
from utils.visualizer import draw_detections

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "violations",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# ---------------------------------------------------------------------------
# Lazy model loading (loaded once per worker process)
# ---------------------------------------------------------------------------

_models: Dict[str, Any] = {}


def _get_models() -> Dict[str, Any]:
    """Load models lazily on first task execution in each worker process."""
    if not _models:
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        _models["custom"] = load_model(os.path.join(model_dir, "best.pt"))
        _models["coco"] = load_coco_model(os.path.join(model_dir, "yolov8n.pt"))
        _models["ocr"] = init_ocr_reader()
    return _models


# ---------------------------------------------------------------------------
# Video processing task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="violations.process_video")
def process_video_task(
    self,
    job_id: str,
    video_path: str,
    confidence: float = 0.25,
) -> Dict[str, Any]:
    """
    Process a video file frame-by-frame for violation detection.

    This mirrors the logic in main.py's _process_video_background but
    uses Celery state updates instead of in-memory job dict.

    Args:
        job_id: Unique identifier for this processing job.
        video_path: Path to the uploaded video file on disk.
        confidence: Detection confidence threshold (0.1 - 0.95).

    Returns:
        Dict with processing results (violations, plates, stats).
    """
    models = _get_models()
    custom_model = models["custom"]
    coco_model = models["coco"]
    ocr_reader = models["ocr"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

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
        persons = detect_persons(detections)
        triple_riding_results = check_triple_riding(motorcycles, persons)
        violation_report = check_violations(detections, triple_riding_results)
        annotated_frame = draw_detections(frame, detections, violation_report, motorcycles)

        out.write(annotated_frame)

        total_detections += len(detections)
        for v in violation_report["violations"]:
            all_violations.add(v)
        max_persons_in_frame = max(
            max_persons_in_frame, violation_report["person_count"]
        )
        total_no_helmets += violation_report["no_helmet_count"]

        # OCR every 10th frame
        if frame_count % 10 == 0:
            for det in detections:
                if det["class_name"] == "license_plate":
                    ocr_result = extract_plate_text(ocr_reader, frame, det["bbox"])
                    if ocr_result["cleaned_text"]:
                        all_plate_texts.append({
                            "text": ocr_result["cleaned_text"],
                            "confidence": ocr_result["confidence"],
                        })

        # Update Celery task state with progress
        progress = frame_count / max(total_frames, 1)
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": min(progress, 1.0),
                "frame": frame_count,
                "total_frames": total_frames,
            },
        )

    cap.release()
    out.release()

    # Deduplicate plate texts
    seen = set()
    unique_plates = []
    for pt in all_plate_texts:
        if pt["text"] not in seen:
            seen.add(pt["text"])
            unique_plates.append(pt)

    # Clean up input video
    try:
        os.unlink(video_path)
    except OSError:
        pass

    return {
        "job_id": job_id,
        "video_path": out_path,
        "total_frames": frame_count,
        "total_detections": total_detections,
        "max_persons_in_frame": max_persons_in_frame,
        "total_no_helmets": total_no_helmets,
        "violations": list(all_violations),
        "unique_plates": unique_plates,
    }
