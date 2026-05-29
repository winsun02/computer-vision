"""Run real-time PPE detection from a webcam."""
from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO

from utils import draw_warning_banner, ensure_file, load_class_names, yolo_result_to_detections
from violation_check import check_ppe_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE from webcam")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "model checkpoint")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index: {args.camera}")

    model = YOLO(str(model_path))
    class_names = load_class_names(model)
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read frame from webcam.")
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detections = yolo_result_to_detections(result, class_names)
            warnings = check_ppe_violations(detections)
            annotated = result.plot()

            frame_count += 1
            fps = frame_count / max(time.time() - start_time, 1e-6)
            cv2.putText(annotated, f"FPS: {fps:.2f}", (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            draw_warning_banner(annotated, warnings)

            cv2.imshow("PPE Detection - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
