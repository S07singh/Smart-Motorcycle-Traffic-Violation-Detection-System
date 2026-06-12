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

# Common OCR misreads for Indian plates — applied in numeric positions
CHAR_CORRECTIONS: Dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
}

# All valid Indian state/UT two-letter codes
VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB",
}

# Known OCR misreads at the state-code position (2 leading chars)
STATE_CODE_CORRECTIONS: Dict[str, str] = {
    "IN": "TN",   # I/T confusion is very common in low-res fonts
    "1N": "TN",
    "UF": "UP",   # P/F confusion
    "U9": "UP",
    "WS": "MS",
    "0D": "OD",
    "KA": "KA",   # already valid — listed for clarity
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
    """Correct small rotations in the plate crop using minAreaRect.

    The original approach of fitting minAreaRect to ALL non-black pixels
    captures the whole image (road photos have almost no pure-black pixels),
    giving a garbage angle.  We now threshold with Otsu first so the rect
    is fitted only to TEXT pixels, and we bail out for large angles which
    indicate the deskew has misread the content.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Otsu binarisation — finds text pixels only, not background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:          # too few text pixels → skip deskew
        return plate_img
    angle = cv2.minAreaRect(coords)[-1]
    # Skip if the angle is implausibly large — deskew has misidentified the text
    if abs(angle) > 15:
        return plate_img
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 1.0:          # negligible tilt → skip the warp
        return plate_img
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


def _correct_state_code(plate: str) -> str:
    """Fix known OCR misreads at the 2-letter state code prefix.

    If the first two characters form an invalid state code and a known
    correction exists, replace them.  This catches "IN" → "TN" etc.
    """
    if len(plate) < 2:
        return plate
    code = plate[:2]
    if code in VALID_STATE_CODES:
        return plate
    correction = STATE_CODE_CORRECTIONS.get(code)
    if correction:
        return correction + plate[2:]
    return plate


def clean_plate_text(raw_text: str) -> str:
    """
    Clean raw OCR output and attempt to match Indian plate formats.

    Supports:
      - Standard:  XX 00 X(X) 0000  (e.g. MH12AB1234)
      - BH-series: 00BH0000X(X)     (e.g. 22BH1234AB)

    For non-Indian plates the raw OCR text is returned as-is (whitespace
    normalised) so the display stays human-readable.
    """
    text = raw_text.upper().strip()
    stripped = re.sub(r"[^A-Z0-9]", "", text)

    # Try standard Indian plate on stripped text
    match = INDIAN_PLATE_PATTERN.search(stripped)
    if match:
        return _correct_state_code(match.group().replace(" ", ""))

    # Try BH-series plate on stripped text
    bh_match = BH_PLATE_PATTERN.search(stripped)
    if bh_match:
        return bh_match.group()

    # Apply OCR character corrections and retry both patterns
    corrected = _apply_char_corrections(stripped)
    match = INDIAN_PLATE_PATTERN.search(corrected)
    if match:
        return _correct_state_code(match.group().replace(" ", ""))

    bh_match = BH_PLATE_PATTERN.search(corrected)
    if bh_match:
        return bh_match.group()

    # No Indian pattern matched — return the whitespace-normalised raw text
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# OCR helper
# ---------------------------------------------------------------------------
def _ocr_attempt(
    reader: PaddleOCR,
    img: np.ndarray,
    det: bool = True,
) -> tuple:
    """Run PaddleOCR on img and return (raw_text, confidence) or None.

    det=True  — use PaddleOCR's text region detector (default).  Good for
                clean plates but may miss characters outside the detected region.
    det=False — skip the region detector and read the entire image as one text
                block.  Catches characters the detector ignores at the edges
                of tight crops (e.g. "UP" and "1234" left off "8AB").
    """
    try:
        results = reader.ocr(img, det=det, cls=True)
        if not results or not results[0]:
            return None
        lines = results[0]
        if det:
            # Format: [[[x1,y1],...], [text, conf]]
            texts = [line[1][0] for line in lines]
            confs = [line[1][1] for line in lines]
        else:
            # Format: [[text, conf], [text, conf], ...]
            items = [item for item in lines if isinstance(item, (list, tuple)) and len(item) >= 2]
            if not items:
                return None
            texts = [item[0] for item in items]
            confs = [item[1] for item in items]
        if not texts:
            return None
        raw = " ".join(texts).strip()
        conf = sum(confs) / len(confs)
        return raw, float(conf)
    except Exception:
        return None



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

    x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2  # save original before padding

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

    # Skip plates that are cut off by the frame edge — when the original bbox
    # (before padding) sits within 15 px of any image border, the plate is
    # partially outside the frame. OCR will only see a clipped fragment
    # (e.g. "02-" instead of "TN-02-AV-6412") and still return a high
    # confidence score for that partial text, which is worse than no result.
    EDGE_MARGIN = 5
    plate_crop = image[y1:y2, x1:x2]
    if (x1_orig < EDGE_MARGIN or x2_orig > w - EDGE_MARGIN
            or y1_orig < EDGE_MARGIN or y2_orig > h - EDGE_MARGIN):
        empty_result["plate_crop"] = plate_crop
        return empty_result
    processed = preprocess_plate(plate_crop)

    # Four strategies — return immediately on the first Indian plate match.
    # Strategy 2 (raw + det=True) is often the best for clean plates because
    # the deskew in preprocessing can still corrupt some crops slightly.
    strategies = [
        (processed,  True),    # 1. preprocessed + region detection
        (plate_crop, True),    # 2. raw crop  + region detection  ← often best
        (processed,  False),   # 3. preprocessed, full image as one text block
        (plate_crop, False),   # 4. raw crop,  full image as one text block
    ]

    best: Dict[str, Any] = {**empty_result, "plate_crop": plate_crop}

    for img_to_ocr, use_det in strategies:
        attempt = _ocr_attempt(reader, img_to_ocr, det=use_det)
        if not attempt:
            continue
        raw, conf = attempt
        cleaned = clean_plate_text(raw)
        # Return immediately on a valid Indian plate match
        stripped = re.sub(r"[^A-Z0-9]", "", cleaned.upper())
        if INDIAN_PLATE_PATTERN.search(stripped) or BH_PLATE_PATTERN.search(stripped):
            return {
                "raw_text": raw,
                "cleaned_text": cleaned,
                "confidence": conf,
                "plate_crop": plate_crop,
            }
        # Keep the longest result as fallback
        if len(cleaned) > len(best.get("cleaned_text", "")):
            best = {
                "raw_text": raw,
                "cleaned_text": cleaned,
                "confidence": conf,
                "plate_crop": plate_crop,
            }

    return best
