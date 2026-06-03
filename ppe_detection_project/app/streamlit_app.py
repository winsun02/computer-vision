"""Simple Streamlit demo for PPE detection on uploaded images."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import detections_from_result, draw_detections, draw_warnings  # noqa: E402
from violation_check import check_ppe_violations, warning_messages  # noqa: E402


st.set_page_config(page_title="PPE Detection Demo", layout="wide")
st.title("Phát hiện và nhận dạng thiết bị bảo hộ lao động")
st.write("Upload một ảnh, chọn model `best.pt`, hệ thống sẽ vẽ bounding box và cảnh báo thiếu PPE.")

model_path = st.sidebar.text_input("Đường dẫn model", "runs/train/ppe_yolo/weights/best.pt")
confidence = st.sidebar.slider("Ngưỡng confidence", 0.05, 0.95, 0.25, 0.05)
image_size = st.sidebar.selectbox("Kích thước ảnh", [320, 416, 512, 640, 768, 1024], index=3)
uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    model_file = (PROJECT_ROOT / model_path).resolve() if not Path(model_path).is_absolute() else Path(model_path)
    if not model_file.is_file():
        st.error(f"Không tìm thấy model: {model_file}")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    image_bgr = cv2.imread(str(temp_path))
    if image_bgr is None:
        st.error("Không thể đọc ảnh upload.")
        st.stop()

    with st.spinner("Đang detect..."):
        model = YOLO(str(model_file))
        result = model.predict(source=image_bgr, conf=confidence, imgsz=image_size, verbose=False)[0]
        detections = detections_from_result(result)
        violations = check_ppe_violations(detections)
        annotated = draw_detections(image_bgr, detections)
        annotated = draw_warnings(annotated, warning_messages(violations))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(Image.open(temp_path), use_container_width=True)
    with col2:
        st.subheader("Kết quả")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Danh sách object phát hiện")
    if detections:
        st.dataframe(
            [
                {"class": det.label, "confidence": round(det.confidence, 4), "box": det.box}
                for det in detections
            ],
            use_container_width=True,
        )
    else:
        st.info("Không phát hiện object nào.")

    st.subheader("Cảnh báo")
    if violations:
        for warning in warning_messages(violations):
            st.error(warning)
    else:
        st.success("Không phát hiện vi phạm theo rule hiện tại.")
else:
    st.info("Hãy upload ảnh để bắt đầu demo.")
