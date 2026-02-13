import cv2
import numpy as np
import easyocr
import re
from typing import Dict, Any, Optional


INDIAN_PLATE_PATTERN = re.compile(r"[A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{4}")


def init_ocr_reader() -> easyocr.Reader:
    reader = easyocr.Reader(["en"], gpu=False)
    return reader


def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
    h, w = plate_img.shape[:2]
    scale_factor = max(2, min(3, 300 // max(w, 1)))
    resized = cv2.resize(
        plate_img,
        (w * scale_factor, h * scale_factor),
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned


def clean_plate_text(raw_text: str) -> str:
    text = raw_text.upper()

    stripped = re.sub(r"[^A-Z0-9]", "", text)

    match = INDIAN_PLATE_PATTERN.search(stripped)
    if match:
        return match.group().replace(" ", "")

    return stripped


def extract_plate_text(
    reader: easyocr.Reader,
    image: np.ndarray,
    bbox: list,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    empty_result = {
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
        ocr_results = reader.readtext(processed, detail=1, paragraph=False)

        if not ocr_results:
            empty_result["plate_crop"] = plate_crop
            return empty_result

        raw_text = " ".join([result[1] for result in ocr_results])
        avg_confidence = sum(result[2] for result in ocr_results) / len(ocr_results)
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
