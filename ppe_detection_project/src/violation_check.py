"""Rule-based PPE violation checking.

The functions in this module are intentionally independent from Ultralytics so
that the logic can be unit-tested or replaced by a more advanced tracker later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_REQUIRED_PPE = ("helmet", "hardhat", "safety vest", "vest")
HELMET_CLASSES = {"helmet", "hardhat"}
NO_HELMET_CLASSES = {"no-helmet", "no helmet", "no_hardhat", "no-hardhat"}
VEST_CLASSES = {"safety vest", "vest", "reflective vest"}
NO_VEST_CLASSES = {"no-vest", "no vest", "no_vest"}
PERSON_CLASSES = {"person", "worker", "human"}


@dataclass(frozen=True)
class Box:
    """Simple bounding box representation using absolute xyxy pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection_area(self, other: "Box") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def region(self, top: float, bottom: float) -> "Box":
        """Return a vertical body region between normalized top/bottom ratios."""
        y1 = self.y1 + self.height * top
        y2 = self.y1 + self.height * bottom
        return Box(self.x1, y1, self.x2, y2)


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _box(det: dict) -> Box:
    x1, y1, x2, y2 = det["bbox"]
    return Box(float(x1), float(y1), float(x2), float(y2))


def _overlap_ratio(candidate: Box, region: Box) -> float:
    """How much of candidate lies inside region."""
    if candidate.area <= 0:
        return 0.0
    return candidate.intersection_area(region) / candidate.area


def check_ppe_violations(
    detections: Iterable[dict],
    helmet_overlap_threshold: float = 0.10,
    vest_overlap_threshold: float = 0.15,
) -> list[str]:
    """Return human-readable PPE violation warnings.

    Logic:
    - If explicit negative classes such as ``no-helmet`` or ``no-vest`` are
      detected, report them immediately.
    - If person/worker boxes exist, verify that helmet-like boxes overlap the
      head region and vest-like boxes overlap the torso region.
    - If the dataset has no person class, still report explicit negative classes.
      When there is PPE but no helmet/vest class, add a global missing warning as
      a conservative demo heuristic.
    """
    detections = list(detections)
    warnings: list[str] = []
    by_name = [(_normalize(det.get("class_name", "")), det) for det in detections]

    if any(name in NO_HELMET_CLASSES for name, _ in by_name):
        warnings.append("NO HELMET")
    if any(name in NO_VEST_CLASSES for name, _ in by_name):
        warnings.append("NO VEST")

    persons = [det for name, det in by_name if name in PERSON_CLASSES]
    helmets = [det for name, det in by_name if name in HELMET_CLASSES]
    vests = [det for name, det in by_name if name in VEST_CLASSES]

    if persons:
        for index, person in enumerate(persons, start=1):
            person_box = _box(person)
            head_region = person_box.region(0.0, 0.30)
            torso_region = person_box.region(0.25, 0.75)

            has_helmet = any(
                _overlap_ratio(_box(helmet), head_region) >= helmet_overlap_threshold
                for helmet in helmets
            )
            has_vest = any(
                _overlap_ratio(_box(vest), torso_region) >= vest_overlap_threshold
                for vest in vests
            )

            suffix = f" (person {index})" if len(persons) > 1 else ""
            if not has_helmet:
                warnings.append(f"NO HELMET{suffix}")
            if not has_vest:
                warnings.append(f"NO VEST{suffix}")
    elif detections:
        # Fallback for PPE-only datasets. This cannot assign violations to a
        # specific person, but it keeps the demo useful when person labels are
        # unavailable.
        if not helmets and not any(name in NO_HELMET_CLASSES for name, _ in by_name):
            warnings.append("NO HELMET (person class unavailable)")
        if not vests and not any(name in NO_VEST_CLASSES for name, _ in by_name):
            warnings.append("NO VEST (person class unavailable)")

    return list(dict.fromkeys(warnings))
