"""Detect PPE trên video."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_detections, draw_warnings, ensure_dir, ensure_file, result_to_detections
from violation_check import check_ppe_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE trên video")
    parser.add_argument("--model", required=True, help="Đường dẫn best.pt")
    parser.add_argument("--source", required=True, help="Đường dẫn video input")
    parser.add_argument("--output", default="runs/detect/video", help="Thư mục lưu video output")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước inference")
    parser.add_argument("--required-ppe", nargs="*", default=["helmet", "vest"], help="PPE bắt buộc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "model")
    video_path = ensure_file(args.source, "video input")
    output_dir = ensure_dir(args.output)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    output_path = output_dir / f"{video_path.stem}_ppe.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    model = YOLO(str(model_path))
    frame_count = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        detections = result_to_detections(result)
        warnings = check_ppe_violations(detections, required_ppe=args.required_ppe)
        output = draw_detections(frame, detections)
        output = draw_warnings(output, warnings)

        frame_count += 1
        elapsed = max(time.time() - start, 1e-6)
        fps = frame_count / elapsed
        cv2.putText(output, f"FPS: {fps:.2f}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        writer.write(output)

    cap.release()
    writer.release()
    print(f"Đã xử lý {frame_count} frame. Lưu video tại: {output_path}")


if __name__ == "__main__":
    main()
