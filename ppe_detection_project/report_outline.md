# Sườn báo cáo bài tập lớn: Phát hiện và nhận dạng thiết bị bảo hộ lao động

## Chương 1: Giới thiệu đề tài

- Bối cảnh an toàn lao động trong nhà máy, công trường, kho vận.
- Nhu cầu tự động phát hiện người lao động không tuân thủ PPE.
- Mục tiêu của đề tài.
- Phạm vi: phát hiện người, helmet, safety vest, mask, gloves, goggles, boots, no-helmet, no-vest.
- Đóng góp chính của hệ thống demo.

## Chương 2: Cơ sở lý thuyết về Object Detection

- Khái niệm object detection.
- Bounding box, class label, confidence score.
- IoU và vai trò trong đánh giá bbox.
- Các hướng tiếp cận one-stage và two-stage detector.
- Các metric: Precision, Recall, F1-score, mAP50, mAP50-95.

## Chương 3: Mô hình YOLO

- Tổng quan họ mô hình YOLO.
- Kiến trúc tổng quát: backbone, neck, head.
- Lý do chọn YOLOv8/YOLO11 cho bài toán PPE.
- Ưu điểm: tốc độ inference nhanh, dễ train, dễ triển khai demo.
- Hạn chế: phụ thuộc dữ liệu, khó phát hiện vật nhỏ nếu ảnh chất lượng thấp.

## Chương 4: Dataset và tiền xử lý dữ liệu

- Nguồn dataset sử dụng và license/citation.
- Các class trong dataset.
- Cấu trúc dữ liệu YOLO: `images/train`, `images/val`, `images/test`, `labels/train`, `labels/val`, `labels/test`.
- Định dạng label YOLO normalized.
- Quy trình kiểm tra và làm sạch annotation.
- Chia train/validation/test.
- Augmentation nếu có.

## Chương 5: Huấn luyện mô hình

- Môi trường thực nghiệm trên Google Colab.
- Cấu hình phần cứng GPU.
- Tham số huấn luyện: model, epochs, imgsz, batch, optimizer nếu có.
- Quy trình train bằng `src/train.py`.
- Theo dõi loss và metric trong thư mục `runs/train`.
- Lưu model `best.pt`.

## Chương 6: Kết quả thực nghiệm

- Kết quả trên validation/test set.
- Bảng Precision, Recall, F1-score, mAP50, mAP50-95.
- Confusion matrix.
- Một số ảnh kết quả minh họa.
- Phân tích lỗi: nhầm helmet/no-helmet, bỏ sót vest, vật nhỏ, che khuất, ánh sáng yếu.

## Chương 7: Demo hệ thống

- Demo detect ảnh đơn bằng `src/detect_image.py`.
- Demo detect video bằng `src/detect_video.py`.
- Demo webcam hoặc video real-time.
- Demo Streamlit upload ảnh.
- Logic cảnh báo trong `src/violation_check.py`.
- Ảnh/video output có bounding box, label, FPS và cảnh báo.

## Chương 8: Kết luận và hướng phát triển

- Kết luận về khả năng áp dụng YOLO cho phát hiện PPE.
- Các kết quả đạt được.
- Hạn chế hiện tại.
- Hướng phát triển: tracking, pose estimation, deploy web/API, tối ưu inference, mở rộng dataset, cảnh báo theo người và theo thời gian.
