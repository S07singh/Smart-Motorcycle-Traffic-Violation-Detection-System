import cv2
import numpy as np
from typing import List, Dict, Any, Set, Tuple

COLORS: Dict[str, tuple] = {
    "helmet": (0, 200, 0),
    "no_helmet": (0, 0, 230),
    "person": (230, 160, 16),
    "person_violation": (0, 0, 230),
    "license_plate": (0, 215, 255),
    "motorcycle": (230, 200, 0),
    "motorcycle_violation": (0, 0, 230),
}

LABEL_DISPLAY: Dict[str, str] = {
    "helmet": "Helmet",
    "no_helmet": "No Helmet",
    "person": "Person",
    "license_plate": "License Plate",
    "motorcycle": "Motorcycle",
}


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    violation_report: Dict[str, Any],
    motorcycles: List[Dict[str, Any]] = None,
) -> np.ndarray:
    annotated = image.copy()

    # Collect bboxes of persons/motorcycles involved in triple riding violations
    # so we only highlight THOSE in red — not every person in the frame
    violating_person_bboxes: Set[Tuple[int, ...]] = violation_report.get(
        "violating_person_bboxes", set()
    )
    violating_mc_bboxes: List[List[int]] = violation_report.get(
        "violating_motorcycle_bboxes", []
    )
    violating_mc_set = {tuple(b) for b in violating_mc_bboxes}

    # Draw motorcycle bounding boxes
    if motorcycles:
        for mc in motorcycles:
            x1, y1, x2, y2 = mc["bbox"]
            mc_tuple = tuple(mc["bbox"])

            if mc_tuple in violating_mc_set:
                colour = COLORS["motorcycle_violation"]
                label_prefix = "⚠ "
            else:
                colour = COLORS["motorcycle"]
                label_prefix = ""

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            label = f"{label_prefix}Motorcycle: {mc['confidence']:.0%}"
            _draw_label(annotated, label, x1, y1, colour)

    # Draw custom model detections
    for det in detections:
        class_name = det["class_name"]
        confidence = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        det_tuple = tuple(det["bbox"])

        # Only persons associated with a violating motorcycle are red
        if class_name == "person" and det_tuple in violating_person_bboxes:
            colour = COLORS["person_violation"]
        else:
            colour = COLORS.get(class_name, (128, 128, 128))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        display_name = LABEL_DISPLAY.get(class_name, class_name)
        label = f"{display_name}: {confidence:.0%}"
        _draw_label(annotated, label, x1, y1, colour)

    return annotated


def _draw_label(image: np.ndarray, label: str, x1: int, y1: int, colour: tuple):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y = max(y1 - 10, text_h + 10)
    cv2.rectangle(
        image,
        (x1, label_y - text_h - 6),
        (x1 + text_w + 6, label_y + 4),
        colour,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, label_y - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
