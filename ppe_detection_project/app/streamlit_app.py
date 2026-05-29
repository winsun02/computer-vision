"""Simple Streamlit demo for PPE detection."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import draw_warning_banner, ensure_file, load_class_names, yolo_result_to_detections  # noqa: E402
from violation_check import check_ppe_violations  # noqa: E402


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def main() -> None:
    st.set_page_config(page_title="PPE Detection", layout="wide")
    st.title("Phát hiện và nhận dạng thiết bị bảo hộ lao động")
    st.write("Upload ảnh, chọn model YOLO đã train và xem cảnh báo thiếu PPE.")

    model_path = st.sidebar.text_input("Đường dẫn model", "runs/train/ppe_yolo/weights/best.pt")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.sidebar.select_slider("Image size", options=[320, 416, 512, 640, 768, 960], value=640)
    uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is None:
        st.info("Vui lòng upload một ảnh để chạy demo.")
        return

    try:
        checked_model_path = ensure_file(model_path, "model checkpoint")
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        image_path = Path(tmp.name)

    try:
        model = load_model(str(checked_model_path))
        result = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        class_names = load_class_names(model)
        detections = yolo_result_to_detections(result, class_names)
        warnings = check_ppe_violations(detections)

        annotated = result.plot()
        draw_warning_banner(annotated, warnings)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        left, right = st.columns([2, 1])
        with left:
            st.image(annotated_rgb, caption="Kết quả detect", use_container_width=True)
        with right:
            st.subheader("Object phát hiện")
            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.write("Không phát hiện object nào.")

            st.subheader("Cảnh báo")
            if warnings:
                for warning in warnings:
                    st.error(warning)
            else:
                st.success("Chưa phát hiện vi phạm PPE theo rule hiện tại.")
    finally:
        image_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
