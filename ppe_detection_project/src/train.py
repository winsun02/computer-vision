"""Train a YOLO PPE detector with Ultralytics."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import ensure_file, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for PPE detection")
    parser.add_argument("--data", default="data/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint, e.g. yolov8n.pt or yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device: '', cpu, 0, 0,1")
    parser.add_argument("--project", default="runs/train", help="Training output directory")
    parser.add_argument("--name", default="ppe_yolo", help="Experiment name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = ensure_file(args.data, "dataset yaml")

    # Ultralytics accepts model aliases that may be downloaded automatically, so
    # only validate local model paths when the argument looks like an existing path.
    model_arg = args.model
    if Path(model_arg).suffix == ".pt" and ("/" in model_arg or "\\" in model_arg):
        model_arg = str(ensure_file(model_arg, "model checkpoint"))

    model = YOLO(model_arg)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(resolve_path(args.project)),
        name=args.name,
    )
    print("Training completed.")
    print(f"Best checkpoint is usually saved at: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
