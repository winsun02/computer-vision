"""Detect PPE real-time bằng webcam.

Lưu ý: Google Colab không hỗ trợ OpenCV VideoCapture webcam như máy local.
File này vẫn được cung cấp để chạy khi notebook/runtime có webcam forwarding hoặc chạy local.
"""
from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO

from utils import draw_detections, draw_warnings, ensure_file, result_to_detections
from violation_check import check_ppe_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE webcam real-time")
    parser.add_argument("--model", required=True, help="Đường dẫn best.pt")
    parser.add_argument("--camera", type=int, default=0, help="ID webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước inference")
    parser.add_argument("--required-ppe", nargs="*", default=["helmet", "vest"], help="PPE bắt buộc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "model")
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam. Trên Google Colab hãy dùng upload video/ảnh hoặc webcam JS notebook.")

    prev_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        detections = result_to_detections(result)
        warnings = check_ppe_violations(detections, required_ppe=args.required_ppe)
        output = draw_detections(frame, detections)
        output = draw_warnings(output, warnings)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(output, f"FPS: {fps:.2f}", (20, output.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("PPE Detection", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
