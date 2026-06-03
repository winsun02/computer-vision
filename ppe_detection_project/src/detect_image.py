"""Run PPE detection on a single image."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .utils import detections_from_result, draw_detections, draw_warnings, ensure_dir, ensure_file, print_detections
    from .violation_check import check_ppe_violations, warning_messages
except ImportError:
    from utils import detections_from_result, draw_detections, draw_warnings, ensure_dir, ensure_file, print_detections
    from violation_check import check_ppe_violations, warning_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE in an image")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--output", default="runs/detect/image", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def run_image_detection(model_path: str | Path, source: str | Path, output_dir: str | Path, conf: float = 0.25, imgsz: int = 640):
    model_path = ensure_file(model_path, "Model weights")
    source_path = ensure_file(source, "Input image")
    output_path = ensure_dir(output_dir)

    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")

    model = YOLO(str(model_path))
    result = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]
    detections = detections_from_result(result)
    violations = check_ppe_violations(detections)

    annotated = draw_detections(image, detections)
    annotated = draw_warnings(annotated, warning_messages(violations))
    saved_path = output_path / f"{source_path.stem}_detected{source_path.suffix}"
    cv2.imwrite(str(saved_path), annotated)
    return saved_path, detections, violations


def main() -> None:
    args = parse_args()
    saved_path, detections, violations = run_image_detection(args.model, args.source, args.output, args.conf, args.imgsz)
    print_detections(detections)
    if violations:
        print("Warnings:")
        for violation in violations:
            print(f"- {violation.message}")
    print(f"Saved result: {saved_path}")


if __name__ == "__main__":
    main()
