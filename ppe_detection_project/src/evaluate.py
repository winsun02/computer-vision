"""Evaluate a trained PPE detection model."""
from __future__ import annotations

import argparse

from ultralytics import YOLO

try:
    from .utils import ensure_file
except ImportError:
    from utils import ensure_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO PPE model")
    parser.add_argument("--model", default="runs/train/ppe_yolo/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--data", default="data/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device: '', cpu, 0, 0,1")
    parser.add_argument("--project", default="runs/evaluate", help="Output project directory")
    parser.add_argument("--name", default="ppe_eval", help="Evaluation run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "Model weights")
    data_path = ensure_file(args.data, "Dataset YAML")
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    print("Evaluation results")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")
    print(f"Artifacts, including confusion matrix when generated, are saved to: {metrics.save_dir}")


if __name__ == "__main__":
    main()
