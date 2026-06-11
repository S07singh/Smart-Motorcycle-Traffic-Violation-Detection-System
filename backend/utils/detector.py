from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any

CLASS_NAMES: Dict[int, str] = {
    0: "helmet",
    1: "no_helmet",
    2: "person",
    3: "license_plate",
}

COCO_MOTORCYCLE_CLASS_ID = 3
COCO_PERSON_CLASS_ID = 0


def load_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    return model


def load_coco_model(model_name: str = "yolov8n.pt") -> YOLO:
    model = YOLO(model_name)
    return model


def run_detection(
    model: YOLO,
    image: np.ndarray,
    confidence: float = 0.25,
) -> List[Dict[str, Any]]:
    results = model(image, conf=confidence, verbose=False)

    detections: List[Dict[str, Any]] = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")

            detections.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

    return detections


def detect_motorcycles(
    coco_model: YOLO,
    image: np.ndarray,
    confidence: float = 0.25,
) -> List[Dict[str, Any]]:
    results = coco_model(image, conf=confidence, classes=[COCO_MOTORCYCLE_CLASS_ID], verbose=False)

    motorcycles: List[Dict[str, Any]] = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())

            motorcycles.append(
                {
                    "class_name": "motorcycle",
                    "class_id": COCO_MOTORCYCLE_CLASS_ID,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

    return motorcycles


def detect_persons_coco(
    coco_model: YOLO,
    image: np.ndarray,
    confidence: float = 0.25,
) -> List[Dict[str, Any]]:
    """Detect full-body persons using the COCO model (class 0).

    Preferred over the custom model for person/rider counting because:
    - Produces full-body bounding boxes (not just head/upper-body)
    - Avoids double-detection of the same rider at different body regions
    - COCO person class is trained on 330k+ diverse images
    """
    results = coco_model(
        image, conf=confidence, classes=[COCO_PERSON_CLASS_ID], verbose=False
    )

    persons: List[Dict[str, Any]] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            persons.append(
                {
                    "class_name": "person",
                    "class_id": COCO_PERSON_CLASS_ID,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

    return persons


def detect_persons(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback: extract persons detected by the custom model."""
    return [det for det in detections if det["class_name"] == "person"]
