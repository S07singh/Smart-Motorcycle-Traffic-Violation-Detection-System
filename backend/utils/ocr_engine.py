"""
OCR Engine for Indian License Plate Recognition.

Uses PaddleOCR v3 with a robust preprocessing pipeline:
  Deskew → Upscale (2-3x bicubic) → Grayscale → CLAHE → Adaptive Threshold → Morph Close → Morph Open

Supports both standard Indian plates (e.g. MH12AB1234) and BH-series plates (e.g. 22BH1234AB).
"""

import cv2
import numpy as np
import re
from typing import Dict, Any
from paddleocr import PaddleOCR

# ---------------------------------------------------------------------------
# Plate patterns
# ---------------------------------------------------------------------------
INDIAN_PLATE_PATTERN = re.compile(r"[A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{4}")
BH_PLATE_PATTERN = re.compile(r"\d{2}BH\d{4}[A-Z]{1,2}")

# Common OCR misreads for Indian plates — applied only in numeric positions
CHAR_CORRECTIONS: Dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_ocr_reader() -> PaddleOCR:
    """Initialise PaddleOCR once (angle classification enabled, CPU-safe)."""
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def deskew_plate(plate_img: np.ndarray) -> np.ndarray:
    """Correct small rotations in the plate crop using minAreaRect."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return plate_img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = plate_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        plate_img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
    """
    Preprocessing pipeline for PaddleOCR:
      1. Deskew (perspective correction)
      2. Upscale 2-3x with bicubic interpolation
      3. Apply CLAHE on each BGR channel for contrast normalisation

    NOTE: We intentionally skip binarisation / thresholding here.
    PaddleOCR's internal detector is trained on real-world colour images.
    Feeding it a binarised (black-and-white) image — even converted back to
    BGR — significantly degrades accuracy because the neural network's feature
    extraction layers rely on colour gradient information that thresholding
    destroys.  CLAHE on the colour image is sufficient to handle dark or
    overexposed plates while keeping all character edge information intact.
    """
    # 1. Deskew
    deskewed = deskew_plate(plate_img)

    # 2. Upscale (2-3x bicubic)
    h, w = deskewed.shape[:2]
    scale_factor = max(2, min(3, 300 // max(w, 1)))
    resized = cv2.resize(
        deskewed,
        (w * scale_factor, h * scale_factor),
        interpolation=cv2.INTER_CUBIC,
    )

    # 3. CLAHE applied per channel in LAB colour space to boost contrast
    #    without distorting hue — keeps the image as a 3-channel BGR array
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_channel, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def _apply_char_corrections(text: str) -> str:
    """Apply character corrections to likely-numeric positions."""
    result = list(text)
    # In standard Indian plates: positions 2-3 are digits, last 4 are digits
    # Apply corrections broadly — the regex match validates structure afterward
    for i, ch in enumerate(result):
        if ch in CHAR_CORRECTIONS:
            result[i] = CHAR_CORRECTIONS[ch]
    return "".join(result)


def clean_plate_text(raw_text: str) -> str:
    """
    Clean raw OCR output and attempt to match Indian plate formats.

    Supports:
      - Standard:  XX 00 X(X) 0000  (e.g. MH12AB1234)
      - BH-series: 00BH0000X(X)     (e.g. 22BH1234AB)

    For non-Indian plates the raw OCR text is returned as-is (whitespace
    normalised) so the display stays human-readable instead of being
    collapsed into a garbled alphanumeric-only string.
    """
    text = raw_text.upper().strip()
    stripped = re.sub(r"[^A-Z0-9]", "", text)

    # Try standard Indian plate on stripped text
    match = INDIAN_PLATE_PATTERN.search(stripped)
    if match:
        return match.group().replace(" ", "")

    # Try BH-series plate on stripped text
    bh_match = BH_PLATE_PATTERN.search(stripped)
    if bh_match:
        return bh_match.group()

    # Apply OCR character corrections and retry both patterns
    corrected = _apply_char_corrections(stripped)
    match = INDIAN_PLATE_PATTERN.search(corrected)
    if match:
        return match.group().replace(" ", "")

    bh_match = BH_PLATE_PATTERN.search(corrected)
    if bh_match:
        return bh_match.group()

    # No Indian pattern matched — return the whitespace-normalised raw text
    # so non-Indian plates remain human-readable (e.g. "29-Y6 4447")
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract_plate_text(
    reader: PaddleOCR,
    image: np.ndarray,
    bbox: list,
) -> Dict[str, Any]:
    """
    Crop the plate region, preprocess, and run PaddleOCR.

    Returns dict with keys: raw_text, cleaned_text, confidence, plate_crop.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Expand the bounding box by 25% on each side — the custom model's plate
    # boxes are often tight, causing OCR to only see the central portion of the
    # plate (e.g., "8AB" instead of "UP 78 AB 1234").
    pad_x = max(6, int((x2 - x1) * 0.25))
    pad_y = max(6, int((y2 - y1) * 0.25))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    empty_result: Dict[str, Any] = {
        "raw_text": "",
        "cleaned_text": "",
        "confidence": 0.0,
        "plate_crop": None,
    }

    if x2 - x1 < 10 or y2 - y1 < 10:
        return empty_result

    plate_crop = image[y1:y2, x1:x2]
    processed = preprocess_plate(plate_crop)

    try:
        ocr_results = reader.ocr(processed, cls=True)

        if not ocr_results or not ocr_results[0]:
            empty_result["plate_crop"] = plate_crop
            return empty_result

        raw_text = " ".join([line[1][0] for line in ocr_results[0]])
        avg_confidence = sum(line[1][1] for line in ocr_results[0]) / len(
            ocr_results[0]
        )
        cleaned_text = clean_plate_text(raw_text)

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "confidence": float(avg_confidence),
            "plate_crop": plate_crop,
        }

    except Exception:
        empty_result["plate_crop"] = plate_crop
        return empty_result
