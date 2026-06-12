from typing import List, Dict, Any, Tuple, Set


def _get_center(bbox: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _point_in_box(point: Tuple[int, int], bbox: List[int]) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _overlap_ratio(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Return what fraction of bbox_a's area overlaps with bbox_b."""
    ix1 = max(bbox_a[0], bbox_b[0])
    iy1 = max(bbox_a[1], bbox_b[1])
    ix2 = min(bbox_a[2], bbox_b[2])
    iy2 = min(bbox_a[3], bbox_b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]), 1)
    return intersection / area_a


def _assign_to_closest_motorcycle(
    det_bbox: List[int],
    motorcycles: List[Dict[str, Any]],
) -> int:
    """Return the index of the motorcycle this detection most likely belongs to.

    Assignment rules (in priority order):
      1. The detection must be in the vertical rider-zone of the motorcycle:
         [mc_y1 - 2 × mc_height, mc_y2].  Detections outside this zone for
         every motorcycle are unassigned (returns -1).
      2. Among eligible motorcycles, pick the one whose horizontal centre is
         closest to the detection's horizontal centre.

    Using closest-motorcycle assignment (instead of a broad overlap check)
    prevents riders on adjacent motorcycles from being double-counted when
    motorcycle bounding boxes or their expanded regions overlap in dense
    traffic scenes.
    """
    d_cx = (_get_center(det_bbox))[0]
    d_cy = (_get_center(det_bbox))[1]

    best_idx = -1
    best_dist = float("inf")

    for i, mc in enumerate(motorcycles):
        mc_x1, mc_y1, mc_x2, mc_y2 = mc["bbox"]
        mc_height = max(mc_y2 - mc_y1, 1)
        mc_width  = max(mc_x2 - mc_x1, 1)
        mc_cx = (mc_x1 + mc_x2) / 2

        # Vertical rider-zone: tightened to 1.2× height above the vehicle.
        # Using 2× was too loose — in dense traffic it pulled riders from
        # adjacent motorcycles into the wrong zone.
        in_y_zone = (mc_y1 - mc_height * 1.2) <= d_cy <= mc_y2
        if not in_y_zone:
            continue

        # Horizontal zone: detection centre must be within the motorcycle's
        # horizontal span with 15% tolerance on each side.
        # Using ±0.8× width-from-center was too loose — in a side-by-side
        # traffic scene a detection 160 px outside the MC was still eligible.
        in_x_zone = (mc_x1 - mc_width * 0.15) <= d_cx <= (mc_x2 + mc_width * 0.15)
        if not in_x_zone:
            continue

        dist = abs(d_cx - mc_cx)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def check_triple_riding(
    motorcycles: List[Dict[str, Any]],
    persons: List[Dict[str, Any]],
    detections: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Check triple riding per motorcycle using exclusive assignment.

    Each helmet/no_helmet detection and each COCO person is assigned to
    exactly ONE motorcycle — the closest one within its vertical rider-zone.
    This prevents double-counting in dense traffic where motorcycle bboxes
    and their expanded regions overlap each other.

    Effective rider count per motorcycle = max(COCO persons, helmet count).
    Triple riding is flagged when effective_count > 2.
    """
    n = len(motorcycles)
    mc_person_lists: List[List[Dict]] = [[] for _ in range(n)]
    mc_helmet_counts: List[int] = [0] * n

    # Assign each COCO person to its closest motorcycle
    for person in persons:
        idx = _assign_to_closest_motorcycle(person["bbox"], motorcycles)
        if idx >= 0:
            mc_person_lists[idx].append(person)

    # Assign each helmet/no_helmet detection to its closest motorcycle
    if detections:
        for det in detections:
            if det["class_name"] not in ("helmet", "no_helmet"):
                continue
            idx = _assign_to_closest_motorcycle(det["bbox"], motorcycles)
            if idx >= 0:
                mc_helmet_counts[idx] += 1

    results: List[Dict[str, Any]] = []
    for i, mc in enumerate(motorcycles):
        associated_persons = mc_person_lists[i]
        face_count = mc_helmet_counts[i]   # helmet + no_helmet from custom model
        coco_count = len(associated_persons)

        # Prefer face-level detections (each = 1 rider on the bike).
        # Fall back to COCO person count ONLY when the custom model found zero
        # faces for this motorcycle (e.g. all riders have helmets that obscure
        # face detection, or camera angle hides faces).
        # We do NOT take max(face, coco) because COCO includes background
        # pedestrians who happen to be closest to this motorcycle.
        effective_count = face_count if face_count > 0 else coco_count
        is_violating = effective_count > 2

        results.append(
            {
                "motorcycle_bbox": mc["bbox"],
                "motorcycle_confidence": mc["confidence"],
                "persons_count": effective_count,
                "person_bboxes": [p["bbox"] for p in associated_persons],
                "is_triple_riding": is_violating,
            }
        )

    return results


def check_violations(
    detections: List[Dict[str, Any]],
    triple_riding_results: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    person_count = 0
    helmet_count = 0
    no_helmet_count = 0
    license_plate_count = 0

    for det in detections:
        cls = det["class_name"]
        if cls == "person":
            person_count += 1
        elif cls == "helmet":
            helmet_count += 1
        elif cls == "no_helmet":
            no_helmet_count += 1
        elif cls == "license_plate":
            license_plate_count += 1

    violations: List[str] = []
    violation_details: List[Dict[str, Any]] = []
    is_triple_riding = False
    violating_person_bboxes: Set[Tuple[int, ...]] = set()
    violating_motorcycle_bboxes: List[List[int]] = []

    if no_helmet_count > 0:
        violations.append("🚫 No Helmet Violation")
        for det in detections:
            if det["class_name"] == "no_helmet":
                violation_details.append(
                    {
                        "violation_type": "No Helmet",
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bbox": det["bbox"],
                    }
                )

    # Per-motorcycle triple riding check
    if triple_riding_results is not None:
        for tr in triple_riding_results:
            if tr["is_triple_riding"]:
                is_triple_riding = True
                violations.append(
                    f"🚫 Triple Riding Violation ({tr['persons_count']} persons on motorcycle)"
                )
                violating_motorcycle_bboxes.append(tr["motorcycle_bbox"])
                for pbbox in tr["person_bboxes"]:
                    violating_person_bboxes.add(tuple(pbbox))
                violation_details.append(
                    {
                        "violation_type": "Triple Riding",
                        "class_name": "motorcycle",
                        "confidence": tr["motorcycle_confidence"],
                        "bbox": tr["motorcycle_bbox"],
                        "persons_count": tr["persons_count"],
                    }
                )
    else:
        # Fallback: global count when no per-motorcycle data is available
        if person_count >= 3:
            is_triple_riding = True
            violations.append(
                f"🚫 Triple Riding Violation ({person_count} persons detected)"
            )
            for det in detections:
                if det["class_name"] == "person":
                    violating_person_bboxes.add(tuple(det["bbox"]))
                    violation_details.append(
                        {
                            "violation_type": "Triple Riding",
                            "class_name": det["class_name"],
                            "confidence": det["confidence"],
                            "bbox": det["bbox"],
                        }
                    )

    motorcycle_count = len(triple_riding_results) if triple_riding_results else 0

    report = {
        "violations": violations,
        "violation_details": violation_details,
        "person_count": person_count,
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "license_plate_count": license_plate_count,
        "motorcycle_count": motorcycle_count,
        "has_helmet": helmet_count > 0,
        "has_no_helmet": no_helmet_count > 0,
        "is_triple_riding": is_triple_riding,
        "violating_person_bboxes": violating_person_bboxes,
        "violating_motorcycle_bboxes": violating_motorcycle_bboxes,
    }

    return report
