from typing import List, Dict, Any, Tuple, Set


def _get_center(bbox: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _point_in_box(point: Tuple[int, int], bbox: List[int]) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def check_triple_riding(
    motorcycles: List[Dict[str, Any]],
    persons: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for mc in motorcycles:
        mc_bbox = mc["bbox"]
        associated_persons = []

        for person in persons:
            person_center = _get_center(person["bbox"])
            if _point_in_box(person_center, mc_bbox):
                associated_persons.append(person)

        is_violating = len(associated_persons) > 2

        results.append(
            {
                "motorcycle_bbox": mc_bbox,
                "motorcycle_confidence": mc["confidence"],
                "persons_count": len(associated_persons),
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

    # Per-motorcycle triple riding check (new logic)
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
        # Fallback: old global count if no motorcycle data available
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
