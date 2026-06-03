"""Streamlit demo cho PPE Detection."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils import draw_detections, draw_warnings, result_to_detections  # noqa: E402
from violation_check import check_ppe_violations  # noqa: E402


st.set_page_config(page_title="PPE Detection Demo", layout="wide")
st.title("Phát hiện và nhận dạng thiết bị bảo hộ lao động")
st.write("Upload ảnh, chọn model YOLO đã train và xem cảnh báo thiếu PPE.")

model_path = st.sidebar.text_input("Đường dẫn model best.pt", value="runs/train/ppe_yolo/weights/best.pt")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Image size", [416, 512, 640, 768, 1024], index=2)
required_ppe = st.sidebar.multiselect(
    "PPE bắt buộc",
    ["helmet", "vest", "mask", "gloves", "goggles", "boots"],
    default=["helmet", "vest"],
)

uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    model_file = Path(model_path)
    if not model_file.exists():
        st.error(f"Không tìm thấy model: {model_file}")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    image_bgr = cv2.imread(str(tmp_path))
    if image_bgr is None:
        st.error("Không đọc được ảnh upload.")
        st.stop()

    with st.spinner("Đang detect..."):
        model = YOLO(str(model_file))
        result = model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
        detections = result_to_detections(result)
        warnings = check_ppe_violations(detections, required_ppe=required_ppe)
        output = draw_detections(image_bgr, detections)
        output = draw_warnings(output, warnings)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(Image.open(tmp_path), use_container_width=True)
    with col2:
        st.subheader("Kết quả")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Object phát hiện")
    if detections:
        st.dataframe(detections, use_container_width=True)
    else:
        st.info("Không phát hiện object nào.")

    st.subheader("Cảnh báo")
    if warnings:
        for warning in warnings:
            st.error(warning)
    else:
        st.success("Không có cảnh báo theo logic hiện tại.")
else:
    st.info("Hãy upload một ảnh để bắt đầu demo.")
