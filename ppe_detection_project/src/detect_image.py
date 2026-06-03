"""Detect PPE trên ảnh đơn."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_detections, draw_warnings, ensure_dir, ensure_file, print_detections, result_to_detections
from violation_check import check_ppe_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE trên ảnh")
    parser.add_argument("--model", required=True, help="Đường dẫn best.pt")
    parser.add_argument("--source", required=True, help="Đường dẫn ảnh input")
    parser.add_argument("--output", default="runs/detect/image", help="Thư mục lưu ảnh output")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước inference")
    parser.add_argument("--required-ppe", nargs="*", default=["helmet", "vest"], help="PPE bắt buộc")
    return parser.parse_args()


def run_image_detection(model_path: str | Path, image_path: str | Path, output_dir: str | Path, conf: float, imgsz: int, required_ppe: list[str]) -> tuple[Path, list[dict], list[str]]:
    model_path = ensure_file(model_path, "model")
    image_path = ensure_file(image_path, "ảnh input")
    output_dir = ensure_dir(output_dir)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    model = YOLO(str(model_path))
    result = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]
    detections = result_to_detections(result)
    warnings = check_ppe_violations(detections, required_ppe=required_ppe)

    output = draw_detections(image, detections)
    output = draw_warnings(output, warnings)
    output_path = output_dir / f"{Path(image_path).stem}_ppe.jpg"
    cv2.imwrite(str(output_path), output)
    return output_path, detections, warnings


def main() -> None:
    args = parse_args()
    output_path, detections, warnings = run_image_detection(args.model, args.source, args.output, args.conf, args.imgsz, args.required_ppe)
    print_detections(detections)
    if warnings:
        print("Cảnh báo:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"Đã lưu ảnh kết quả: {output_path}")


if __name__ == "__main__":
    main()
