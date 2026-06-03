"""Train YOLO model cho PPE Detection."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import ensure_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO cho phát hiện PPE")
    parser.add_argument("--data", default="data/data.yaml", help="Đường dẫn data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Model khởi tạo, ví dụ yolov8n.pt/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh train")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Thiết bị: 0, cpu, cuda:0")
    parser.add_argument("--project", default="runs/train", help="Thư mục lưu kết quả")
    parser.add_argument("--name", default="ppe_yolo", help="Tên experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = ensure_file(args.data, "data.yaml")

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
    print(f"Train xong. best.pt thường nằm tại: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
