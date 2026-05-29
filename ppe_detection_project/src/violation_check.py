"""PPE violation checking logic.

The heuristics here are intentionally isolated so you can replace them with
site-specific rules without touching training or inference code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .utils import Detection, box_center, filter_detections, is_label, point_in_box, region_box
except ImportError:  # Allows running this file directly with `python src/violation_check.py`.
    from utils import Detection, box_center, filter_detections, is_label, point_in_box, region_box


@dataclass(frozen=True)
class Violation:
    """A PPE violation warning."""

    message: str
    person_box: tuple[int, int, int, int] | None = None


DEFAULT_REQUIRED_PPE = ("helmet", "safety_vest")
PPE_REGION = {
    "helmet": "head",
    "safety_vest": "torso",
    "mask": "head",
    "no_mask": "head",
    "goggles": "head",
    "gloves": "hands",
    "boots": "feet",
}
EXPLICIT_NEGATIVE = {
    "helmet": "no_helmet",
    "safety_vest": "no_vest",
    "mask": "no_mask",
}
WARNING_TEXT = {
    "helmet": "NO HELMET",
    "safety_vest": "NO VEST",
    "mask": "NO MASK",
    "gloves": "NO GLOVES",
    "goggles": "NO GOGGLES",
    "boots": "NO BOOTS",
}


def _has_ppe_in_region(person_box: tuple[int, int, int, int], detections: Iterable[Detection], ppe: str) -> bool:
    """Check whether a PPE detection center falls in the expected body region."""
    expected_region = region_box(person_box, PPE_REGION.get(ppe, "full"))
    return any(point_in_box(box_center(det.box), expected_region) for det in filter_detections(detections, ppe))


def _has_explicit_negative(person_box: tuple[int, int, int, int], detections: Iterable[Detection], ppe: str) -> bool:
    """Check dataset labels such as no_helmet/no_vest inside a person box."""
    negative_class = EXPLICIT_NEGATIVE.get(ppe)
    if not negative_class:
        return False
    return any(point_in_box(box_center(det.box), person_box) for det in filter_detections(detections, negative_class))


def check_ppe_violations(
    detections: Iterable[Detection],
    required_ppe: Iterable[str] = DEFAULT_REQUIRED_PPE,
) -> list[Violation]:
    """Return PPE violations for a frame/image.

    If a `person` class exists, each person is checked independently. If the
    dataset does not include `person`, the function still reports explicit
    negative classes (`no_helmet`, `no_vest`) and a conservative global warning
    when PPE objects are present but a required class is absent.
    """
    detections = list(detections)
    required = [item for item in required_ppe if item]
    persons = filter_detections(detections, "person")
    violations: list[Violation] = []

    if persons:
        for index, person in enumerate(persons, start=1):
            for ppe in required:
                if _has_explicit_negative(person.box, detections, ppe) or not _has_ppe_in_region(person.box, detections, ppe):
                    violations.append(Violation(f"Person {index}: {WARNING_TEXT.get(ppe, f'NO {ppe.upper()}')}", person.box))
        return violations

    # Fallback for datasets without a person class.
    for ppe in required:
        negative_class = EXPLICIT_NEGATIVE.get(ppe)
        if negative_class and any(is_label(det.label, negative_class) for det in detections):
            violations.append(Violation(WARNING_TEXT.get(ppe, f"NO {ppe.upper()}")))

    has_any_ppe = any(any(is_label(det.label, ppe) for ppe in PPE_REGION) for det in detections)
    if has_any_ppe:
        for ppe in required:
            if not any(is_label(det.label, ppe) for det in detections):
                violations.append(Violation(f"GLOBAL CHECK: {WARNING_TEXT.get(ppe, f'NO {ppe.upper()}')}"))

    return violations


def warning_messages(violations: Iterable[Violation]) -> list[str]:
    """Convert violations to drawable warning strings."""
    return [violation.message for violation in violations]
