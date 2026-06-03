"""Run PPE detection on a video file."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .utils import detections_from_result, draw_detections, draw_label, draw_warnings, ensure_dir, ensure_file
    from .violation_check import check_ppe_violations, warning_messages
except ImportError:
    from utils import detections_from_result, draw_detections, draw_label, draw_warnings, ensure_dir, ensure_file
    from violation_check import check_ppe_violations, warning_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE in a video")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default="runs/detect/video", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    return parser.parse_args()


def run_video_detection(model_path: str | Path, source: str | Path, output_dir: str | Path, conf: float = 0.25, imgsz: int = 640, show: bool = False) -> Path:
    model_path = ensure_file(model_path, "Model weights")
    source_path = ensure_file(source, "Input video")
    output_path = ensure_dir(output_dir)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saved_path = output_path / f"{source_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(saved_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))
    model = YOLO(str(model_path))

    prev_time = time.perf_counter()
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            detections = detections_from_result(result)
            violations = check_ppe_violations(detections)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            annotated = draw_detections(frame, detections)
            annotated = draw_warnings(annotated, warning_messages(violations))
            draw_label(annotated, f"FPS: {fps:.1f}", (10, height - 15), (255, 128, 0))
            writer.write(annotated)
            frame_count += 1

            if show:
                cv2.imshow("PPE Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames. Saved result: {saved_path}")
    return saved_path


def main() -> None:
    args = parse_args()
    run_video_detection(args.model, args.source, args.output, args.conf, args.imgsz, args.show)


if __name__ == "__main__":
    main()
