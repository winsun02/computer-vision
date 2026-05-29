"""Run PPE detection on one image and save an annotated result."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_warning_banner, ensure_dir, ensure_file, load_class_names, print_detections, yolo_result_to_detections
from violation_check import check_ppe_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE in a single image")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--output", default="runs/detect/image", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "model checkpoint")
    image_path = ensure_file(args.source, "input image")
    output_dir = ensure_dir(args.output)

    model = YOLO(str(model_path))
    result = model.predict(str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    class_names = load_class_names(model)
    detections = yolo_result_to_detections(result, class_names)
    warnings = check_ppe_violations(detections)

    annotated = result.plot()
    annotated = draw_warning_banner(annotated, warnings)
    output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated)

    print_detections(detections)
    if warnings:
        print("Warnings:", "; ".join(warnings))
    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()
