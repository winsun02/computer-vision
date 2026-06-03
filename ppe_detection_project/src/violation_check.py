"""Logic kiểm tra vi phạm thiết bị bảo hộ lao động."""
from __future__ import annotations

from typing import Sequence

from utils import canonical_label


def _center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _inside_region(box: Sequence[float], region: Sequence[float]) -> bool:
    cx, cy = _center(box)
    x1, y1, x2, y2 = region
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _person_regions(person_box: Sequence[float]) -> dict[str, list[float]]:
    """Chia bbox người thành các vùng tương đối: đầu, thân, tay, chân."""
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1
    return {
        "head": [x1 + 0.15 * w, y1, x2 - 0.15 * w, y1 + 0.30 * h],
        "torso": [x1 + 0.10 * w, y1 + 0.25 * h, x2 - 0.10 * w, y1 + 0.70 * h],
        "hands": [x1, y1 + 0.30 * h, x2, y1 + 0.78 * h],
        "feet": [x1 + 0.05 * w, y1 + 0.72 * h, x2 - 0.05 * w, y2],
    }


def check_ppe_violations(
    detections: Sequence[dict],
    required_ppe: Sequence[str] = ("helmet", "vest"),
    use_region_check: bool = True,
) -> list[str]:
    """
    Kiểm tra vi phạm PPE từ danh sách detection.

    - Nếu có class `person`, kiểm tra PPE có nằm trong vùng phù hợp của từng người không.
    - Nếu không có class `person`, vẫn cảnh báo dựa trên class phủ định như `no_helmet`, `no_vest`.
    - `required_ppe` có thể mở rộng: helmet, vest, mask, gloves, goggles, boots.
    """
    canonical_dets = [{**det, "canonical_label": canonical_label(det.get("canonical_label") or det["label"])} for det in detections]
    persons = [det for det in canonical_dets if det["canonical_label"] == "person"]
    warnings: list[str] = []

    negative_map = {
        "no_helmet": "NO HELMET",
        "no_vest": "NO VEST",
    }
    for det in canonical_dets:
        if det["canonical_label"] in negative_map:
            msg = negative_map[det["canonical_label"]]
            if msg not in warnings:
                warnings.append(msg)

    if not persons:
        # Dataset không có person: không thể gán PPE theo từng người một cách chắc chắn.
        # Vẫn trả về cảnh báo từ class phủ định nếu model học các class no_*.
        return warnings

    for person_idx, person in enumerate(persons, start=1):
        regions = _person_regions(person["box"])
        for ppe in required_ppe:
            ppe = canonical_label(ppe)
            if ppe == "helmet":
                matched = any(
                    det["canonical_label"] == "helmet" and (not use_region_check or _inside_region(det["box"], regions["head"]))
                    for det in canonical_dets
                )
                if not matched:
                    warnings.append(f"PERSON {person_idx}: NO HELMET")
            elif ppe == "vest":
                matched = any(
                    det["canonical_label"] == "vest" and (not use_region_check or _inside_region(det["box"], regions["torso"]))
                    for det in canonical_dets
                )
                if not matched:
                    warnings.append(f"PERSON {person_idx}: NO VEST")
            elif ppe == "mask":
                matched = any(det["canonical_label"] == "mask" and _inside_region(det["box"], regions["head"]) for det in canonical_dets)
                if not matched:
                    warnings.append(f"PERSON {person_idx}: NO MASK")
            elif ppe == "gloves":
                matched = any(det["canonical_label"] == "gloves" and _inside_region(det["box"], regions["hands"]) for det in canonical_dets)
                if not matched:
                    warnings.append(f"PERSON {person_idx}: NO GLOVES")
            elif ppe == "goggles":
                matched = any(det["canonical_label"] == "goggles" and _inside_region(det["box"], regions["head"]) for det in canonical_dets)
                if not matched:
                    warnings.append(f"PERSON {person_idx}: NO GOGGLES")
            elif ppe == "boots":
                matched = any(det["canonical_label"] == "boots" and _inside_region(det["box"], regions["feet"]) for det in canonical_dets)
                if not matched:
                    warnings.append(f"PERSON {person_idx}: NO BOOTS")
    return warnings
