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
    Full preprocessing pipeline:
      1. Deskew (perspective correction)
      2. Upscale 2-3x with bicubic interpolation
      3. Convert to grayscale
      4. CLAHE for contrast normalisation (handles dark / overexposed plates)
      5. Single adaptive threshold (Gaussian)
      6. Morphological close (fill small gaps)
      7. Morphological open (remove noise)
    """
    # 1. Deskew
    deskewed = deskew_plate(plate_img)

    # 2. Upscale
    h, w = deskewed.shape[:2]
    scale_factor = max(2, min(3, 300 // max(w, 1)))
    resized = cv2.resize(
        deskewed,
        (w * scale_factor, h * scale_factor),
        interpolation=cv2.INTER_CUBIC,
    )

    # 3. Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5. Single adaptive threshold (bug fix: removed duplicate call)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # 6 & 7. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned


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
    """
    text = raw_text.upper()
    stripped = re.sub(r"[^A-Z0-9]", "", text)

    # Try standard Indian plate
    match = INDIAN_PLATE_PATTERN.search(stripped)
    if match:
        result = match.group().replace(" ", "")
        return result

    # Try BH-series plate
    bh_match = BH_PLATE_PATTERN.search(stripped)
    if bh_match:
        return bh_match.group()

    # Fallback: apply corrections and retry
    corrected = _apply_char_corrections(stripped)
    match = INDIAN_PLATE_PATTERN.search(corrected)
    if match:
        return match.group().replace(" ", "")

    bh_match = BH_PLATE_PATTERN.search(corrected)
    if bh_match:
        return bh_match.group()

    return stripped


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

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

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
