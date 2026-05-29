"""Evaluate a trained YOLO PPE model."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import ensure_file, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPE detector")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--data", default="data/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device: '', cpu, 0, 0,1")
    parser.add_argument("--project", default="runs/evaluate", help="Evaluation output directory")
    parser.add_argument("--name", default="ppe_eval", help="Experiment name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "model checkpoint")
    data_path = ensure_file(args.data, "dataset yaml")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        project=str(resolve_path(args.project)),
        name=args.name,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    print("Evaluation results")
    print(f"Split      : {args.split}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"mAP50      : {map50:.4f}")
    print(f"mAP50-95   : {map5095:.4f}")
    print(f"Artifacts  : {Path(metrics.save_dir)}")
    print("If plots=True is supported by your Ultralytics version, confusion_matrix.png is saved in the artifacts folder.")


if __name__ == "__main__":
    main()
