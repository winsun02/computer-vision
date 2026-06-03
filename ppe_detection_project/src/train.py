"""Train a YOLO model for PPE detection."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

try:
    from .utils import ensure_file
except ImportError:
    from utils import ensure_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for PPE detection")
    parser.add_argument("--data", default="data/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint, e.g. yolov8n.pt or yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device: '', cpu, 0, 0,1")
    parser.add_argument("--project", default="runs/train", help="Output project directory")
    parser.add_argument("--name", default="ppe_yolo", help="Experiment name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = ensure_file(args.data, "Dataset YAML")
    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    save_dir = Path(results.save_dir)
    print(f"Training finished. Best weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
