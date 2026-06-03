"""Run real-time PPE detection from a webcam."""
from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO

try:
    from .utils import detections_from_result, draw_detections, draw_label, draw_warnings, ensure_file
    from .violation_check import check_ppe_violations, warning_messages
except ImportError:
    from utils import detections_from_result, draw_detections, draw_label, draw_warnings, ensure_file
    from violation_check import check_ppe_violations, warning_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE with a webcam")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "Model weights")
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise ValueError(f"Could not open webcam index {args.camera}")

    prev_time = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break
            result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detections = detections_from_result(result)
            violations = check_ppe_violations(detections)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            annotated = draw_detections(frame, detections)
            annotated = draw_warnings(annotated, warning_messages(violations))
            draw_label(annotated, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15), (255, 128, 0))
            cv2.imshow("PPE Detection - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
