"""Đánh giá YOLO model trên validation/test set."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import ensure_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPE YOLO model")
    parser.add_argument("--model", required=True, help="Đường dẫn best.pt")
    parser.add_argument("--data", default="data/data.yaml", help="Đường dẫn data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Tập đánh giá")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Thiết bị: 0/cpu")
    parser.add_argument("--project", default="runs/evaluate", help="Thư mục lưu kết quả")
    parser.add_argument("--name", default="ppe_eval", help="Tên experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_file(args.model, "model")
    data_path = ensure_file(args.data, "data.yaml")

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

    print("Kết quả đánh giá:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map5095:.4f}")
    print(f"Confusion matrix/plots nếu có được lưu tại: {Path(metrics.save_dir)}")


if __name__ == "__main__":
    main()
