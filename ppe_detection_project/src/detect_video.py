"""Run PPE detection on a video file and save an annotated video."""
from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO

from utils import draw_warning_banner, ensure_dir, ensure_file, load_class_names, open_video_writer, yolo_result_to_detections
from violation_check import check_ppe_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PPE in video")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default="runs/detect/video", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--show", action="store_true", help="Show frames while processing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "model checkpoint")
    video_path = ensure_file(args.source, "input video")
    output_dir = ensure_dir(args.output)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir / f"{video_path.stem}_detected.mp4"
    writer = open_video_writer(output_path, fps, (width, height))

    model = YOLO(str(model_path))
    class_names = load_class_names(model)
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detections = yolo_result_to_detections(result, class_names)
            warnings = check_ppe_violations(detections)
            annotated = result.plot()

            frame_count += 1
            elapsed = max(time.time() - start_time, 1e-6)
            current_fps = frame_count / elapsed
            cv2.putText(annotated, f"FPS: {current_fps:.2f}", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            draw_warning_banner(annotated, warnings)
            writer.write(annotated)

            if args.show:
                cv2.imshow("PPE Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames. Saved result: {output_path}")


if __name__ == "__main__":
    main()
