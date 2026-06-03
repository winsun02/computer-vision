"""Tiện ích dùng chung cho project PPE Detection trên Google Colab."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from ultralytics.engine.results import Results


PPE_ALIASES = {
    "helmet": {"helmet", "hardhat", "hard_hat", "hard-hat", "safety_helmet", "safety helmet"},
    "vest": {"vest", "safety_vest", "safety vest", "reflective_vest", "jacket"},
    "mask": {"mask", "face_mask", "face mask", "respirator"},
    "gloves": {"gloves", "glove", "safety_gloves"},
    "goggles": {"goggles", "glasses", "safety_goggles", "safety glasses"},
    "boots": {"boots", "boot", "safety_boots", "shoes", "safety_shoes"},
    "person": {"person", "worker", "human"},
    "no_helmet": {"no_helmet", "no-helmet", "without_helmet", "no hardhat", "no_hardhat"},
    "no_vest": {"no_vest", "no-vest", "without_vest", "no safety vest"},
}


def ensure_file(path: str | Path, description: str = "file") -> Path:
    """Kiểm tra đường dẫn file tồn tại."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {description}: {file_path}")
    return file_path


def ensure_dir(path: str | Path) -> Path:
    """Tạo thư mục nếu chưa tồn tại và trả về Path."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def normalize_name(name: str) -> str:
    """Chuẩn hóa tên class để so khớp linh hoạt."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_label(name: str) -> str:
    """Ánh xạ tên class trong dataset về nhãn PPE chuẩn nếu có thể."""
    normalized = normalize_name(name)
    for canonical, aliases in PPE_ALIASES.items():
        normalized_aliases = {normalize_name(alias) for alias in aliases}
        if normalized in normalized_aliases:
            return canonical
    return normalized


def get_class_name(names: dict | Sequence[str], class_id: int) -> str:
    """Lấy tên class từ metadata của Ultralytics."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def result_to_detections(result: Results) -> List[dict]:
    """Chuyển Ultralytics Results thành list dict dễ xử lý."""
    detections: List[dict] = []
    names = result.names
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, class_id in zip(boxes, confs, classes):
        label = get_class_name(names, int(class_id))
        detections.append(
            {
                "class_id": int(class_id),
                "label": label,
                "canonical_label": canonical_label(label),
                "confidence": float(conf),
                "box": [float(v) for v in box],
            }
        )
    return detections


def draw_detections(image: np.ndarray, detections: Iterable[dict]) -> np.ndarray:
    """Vẽ bounding box và label lên ảnh bằng OpenCV."""
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = det["label"]
        conf = det["confidence"]
        color = (0, 180, 0)
        if det.get("canonical_label") in {"no_helmet", "no_vest"}:
            color = (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(output, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


def draw_warnings(image: np.ndarray, warnings: Sequence[str]) -> np.ndarray:
    """Hiển thị cảnh báo PPE ở góc trên ảnh/frame."""
    output = image.copy()
    if not warnings:
        return output

    cv2.rectangle(output, (10, 10), (min(output.shape[1] - 10, 760), 45 + 32 * len(warnings)), (0, 0, 255), -1)
    for idx, warning in enumerate(warnings):
        cv2.putText(
            output,
            warning,
            (20, 40 + idx * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
    return output


def print_detections(detections: Sequence[dict]) -> None:
    """In danh sách object đã phát hiện."""
    if not detections:
        print("Không phát hiện object nào.")
        return
    print("Danh sách object phát hiện:")
    for idx, det in enumerate(detections, start=1):
        box = ", ".join(f"{v:.1f}" for v in det["box"])
        print(f"{idx:02d}. {det['label']} | conf={det['confidence']:.3f} | box=[{box}]")
