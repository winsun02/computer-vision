"""Utility functions for the PPE detection project."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2


def resolve_path(path: str | Path) -> Path:
    """Return a normalized Path without requiring it to exist."""
    return Path(path).expanduser().resolve()


def ensure_file(path: str | Path, description: str = "file") -> Path:
    """Validate that a path exists and is a file."""
    file_path = resolve_path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"{description.capitalize()} not found: {file_path}")
    return file_path


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    dir_path = resolve_path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_class_names(model) -> dict[int, str]:
    """Read class names from an Ultralytics model in a stable dict format."""
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def yolo_result_to_detections(result, class_names: dict[int, str]) -> list[dict]:
    """Convert one Ultralytics result object to serializable detections."""
    detections: list[dict] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        cls_id = int(box.cls.item())
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf.item()),
                "bbox": xyxy,
            }
        )
    return detections


def print_detections(detections: Iterable[dict]) -> None:
    """Print detections in a readable tabular text format."""
    detections = list(detections)
    if not detections:
        print("No objects detected.")
        return

    print("Detected objects:")
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        print(
            f"{idx:02d}. {det['class_name']} "
            f"conf={det['confidence']:.3f} "
            f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )


def draw_warning_banner(image, warnings: Iterable[str], color=(0, 0, 255)):
    """Draw violation warnings on an OpenCV BGR image."""
    warnings = list(dict.fromkeys(warnings))
    if not warnings:
        return image

    banner_text = " | ".join(warnings)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(banner_text, font, scale, thickness)
    cv2.rectangle(image, (0, 0), (min(image.shape[1], text_w + 20), text_h + 22), color, -1)
    cv2.putText(image, banner_text, (10, text_h + 10), font, scale, (255, 255, 255), thickness)
    return image


def open_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]):
    """Create a cross-platform OpenCV VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
