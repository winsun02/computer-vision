# Sườn báo cáo bài tập lớn: Phát hiện và nhận dạng thiết bị bảo hộ lao động

## Chương 1: Giới thiệu đề tài
- Bối cảnh an toàn lao động trong nhà máy/công trường.
- Bài toán phát hiện thiết bị bảo hộ cá nhân (PPE).
- Mục tiêu, phạm vi và đóng góp của đề tài.

## Chương 2: Cơ sở lý thuyết về Object Detection
- Khái niệm object detection, bounding box, class label, confidence score.
- So sánh one-stage detector và two-stage detector.
- Các metric đánh giá: IoU, Precision, Recall, F1-score, AP, mAP.

## Chương 3: Mô hình YOLO
- Lịch sử phát triển YOLO.
- Kiến trúc tổng quát của YOLO.
- Lý do chọn YOLOv8/YOLOv11 cho demo thực tế.
- Quy trình inference và Non-Maximum Suppression (NMS).

## Chương 4: Dataset và tiền xử lý dữ liệu
- Dataset sử dụng: Construction Site Safety Image Dataset Roboflow trên Kaggle.
- Lý do chọn dataset: đúng bối cảnh công trường, có person, Hardhat, Safety Vest, Mask và các class vi phạm.
- Class gốc và class chuẩn hóa: Hardhat → helmet, NO-Hardhat → no_helmet, Safety Vest → safety_vest, NO-Safety Vest → no_vest, ...
- Cấu trúc dataset YOLO sau chuẩn hóa: `images/train`, `images/val`, `images/test`, `labels/train`, `labels/val`, `labels/test`.
- Quy trình dùng `src/prepare_css_dataset.py` để download/copy/symlink ảnh và remap label.
- Phạm vi xử lý class: chỉ dùng các class có trong dataset đã chọn, không thêm class ngoài dataset.
- Kiểm tra chất lượng nhãn và chia train/val/test.
- Augmentation nếu có.

## Chương 5: Huấn luyện mô hình
- Cấu hình môi trường phần cứng/phần mềm.
- Tham số huấn luyện: model, epochs, imgsz, batch, optimizer.
- Cách chạy script `src/train.py` trên máy local và trên Google Colab bằng `notebooks/ppe_detection_colab.ipynb`.
- Theo dõi loss và chọn `best.pt`.

## Chương 6: Kết quả thực nghiệm
- Bảng kết quả Precision, Recall, F1-score, mAP50, mAP50-95.
- Confusion matrix.
- Ví dụ ảnh đúng/sai.
- Phân tích lỗi thường gặp: che khuất, ánh sáng yếu, PPE nhỏ, nhãn thiếu.

## Chương 7: Demo hệ thống
- Kiến trúc demo ảnh/video/webcam.
- Luồng xử lý: đọc input → YOLO inference → kiểm tra vi phạm → vẽ kết quả.
- Demo Streamlit upload ảnh và notebook Google Colab cho train/evaluate/inference.
- Ví dụ cảnh báo NO HELMET, NO VEST.

## Chương 8: Kết luận và hướng phát triển
- Kết quả đạt được.
- Hạn chế của hệ thống hiện tại.
- Hướng phát triển: tracking theo người, thêm class, tối ưu TensorRT/ONNX, triển khai edge camera.
