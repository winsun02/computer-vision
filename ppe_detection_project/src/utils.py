"""Shared helpers for the PPE detection project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Detection:
    """A normalized object detection returned by YOLO."""

    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


# Common aliases make the violation logic resilient to different datasets.
CLASS_ALIASES: dict[str, set[str]] = {
    "helmet": {"helmet", "hardhat", "hard_hat", "hard-hat"},
    "mask": {"mask"},
    "no_helmet": {"no_helmet", "no-helmet", "no_hardhat", "no-hardhat"},
    "no_mask": {"no_mask", "no-mask"},
    "no_vest": {"no_vest", "no-vest", "no_safety_vest", "no-safety-vest"},
    "person": {"person"},
    "safety_cone": {"safety_cone", "safety-cone"},
    "safety_vest": {"safety_vest", "safety-vest", "vest"},
    "machinery": {"machinery"},
    "vehicle": {"vehicle"},
}


def resolve_path(path: str | Path) -> Path:
    """Resolve a user supplied path without requiring it to already exist."""
    return Path(path).expanduser().resolve()


def ensure_file(path: str | Path, description: str) -> Path:
    """Validate that a path exists and is a file."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file."""
    yaml_path = ensure_file(path, "YAML file")
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_label(label: str) -> str:
    """Normalize class labels for robust matching."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def is_label(label: str, canonical: str) -> bool:
    """Return True when `label` matches a canonical class or one of its aliases."""
    normalized = normalize_label(label)
    aliases = {normalize_label(item) for item in CLASS_ALIASES.get(canonical, {canonical})}
    return normalized in aliases


def filter_detections(detections: Iterable[Detection], canonical: str) -> list[Detection]:
    """Filter detections by canonical class name using aliases."""
    return [det for det in detections if is_label(det.label, canonical)]


def detections_from_result(result) -> list[Detection]:
    """Convert an Ultralytics result object to a list of Detection objects."""
    detections: list[Detection] = []
    names = result.names
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        detections.append(
            Detection(
                class_id=class_id,
                label=str(names.get(class_id, class_id)),
                confidence=confidence,
                box=(x1, y1, x2, y2),
            )
        )
    return detections


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    """Return True when a point lies inside a bounding box."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def region_box(person_box: tuple[int, int, int, int], region: str) -> tuple[int, int, int, int]:
    """Return an approximate body region inside a person box."""
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    if region == "head":
        return x1, y1, x2, y1 + int(0.35 * height)
    if region == "torso":
        return x1, y1 + int(0.25 * height), x2, y1 + int(0.75 * height)
    return person_box


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Draw readable text with a filled background."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(y, height + baseline + 4)
    cv2.rectangle(image, (x, y - height - baseline - 4), (x + width + 4, y + 4), color, -1)
    cv2.putText(image, text, (x + 2, y - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detections(image: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels for detections."""
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = (0, 0, 255) if det.label.startswith("no") else (0, 180, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        draw_label(output, f"{det.label} {det.confidence:.2f}", (x1, y1 - 6), color)
    return output


def draw_warnings(image: np.ndarray, warnings: Iterable[str]) -> np.ndarray:
    """Draw violation warnings at the top-left corner."""
    output = image.copy()
    for index, warning in enumerate(warnings):
        draw_label(output, warning, (10, 25 + index * 32), (0, 0, 255), font_scale=0.7, thickness=2)
    return output


def print_detections(detections: Iterable[Detection]) -> None:
    """Print detections as a compact table."""
    print("Detected objects:")
    for det in detections:
        print(f"- {det.label:<15} conf={det.confidence:.3f} box={det.box}")
