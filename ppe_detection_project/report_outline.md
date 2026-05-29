# Sườn báo cáo: Phát hiện và nhận dạng thiết bị bảo hộ lao động trong ảnh/video

## Chương 1: Giới thiệu đề tài
- Bối cảnh an toàn lao động trong nhà máy, công trường, kho vận.
- Vấn đề cần giải quyết: giám sát việc sử dụng PPE bằng thị giác máy tính.
- Mục tiêu của đề tài.
- Phạm vi: ảnh, video, webcam real-time; các lớp PPE chính.
- Đóng góp chính của project.

## Chương 2: Cơ sở lý thuyết về Object Detection
- Khái niệm object detection.
- Bounding box, class label, confidence score.
- Bài toán one-stage detector và two-stage detector.
- IoU và Non-Maximum Suppression.
- Các metric đánh giá: Precision, Recall, F1-score, AP, mAP.

## Chương 3: Mô hình YOLO
- Lịch sử phát triển YOLO.
- Nguyên lý hoạt động tổng quát của YOLO.
- Lý do chọn YOLOv8/YOLOv11 qua thư viện Ultralytics.
- Ưu điểm khi triển khai demo real-time.
- Các tham số huấn luyện quan trọng: model, epochs, imgsz, batch, device.

## Chương 4: Dataset và tiền xử lý dữ liệu
- Mô tả nguồn dataset sử dụng, giấy phép và số lượng ảnh.
- Danh sách class: person, helmet, safety vest, mask, gloves, goggles, boots, no-helmet, no-vest.
- Cấu trúc YOLO: images/train, images/val, images/test, labels/train, labels/val, labels/test.
- Định dạng nhãn YOLO: `class_id x_center y_center width height` đã chuẩn hóa [0, 1].
- Quy trình gán nhãn/kiểm tra nhãn.
- Chia tập train/val/test.
- Tiền xử lý và augmentation.

## Chương 5: Huấn luyện mô hình
- Môi trường phần cứng/phần mềm.
- Cài đặt thư viện.
- File cấu hình `data/data.yaml`.
- Lệnh train và ý nghĩa tham số.
- Theo dõi loss và lưu checkpoint `best.pt`.

## Chương 6: Kết quả thực nghiệm
- Bảng kết quả Precision, Recall, F1-score, mAP50, mAP50-95.
- Confusion matrix.
- Một số ảnh kết quả đúng.
- Một số lỗi thường gặp: nhầm class, thiếu object nhỏ, che khuất, ánh sáng yếu.
- Phân tích nguyên nhân và đề xuất cải thiện.

## Chương 7: Demo hệ thống
- Kiến trúc demo: CLI detect ảnh, CLI detect video, webcam, Streamlit.
- Luồng xử lý: input -> YOLO inference -> rule kiểm tra vi phạm -> output có box/label/cảnh báo.
- Minh họa giao diện Streamlit.
- Tốc độ xử lý FPS.

## Chương 8: Kết luận và hướng phát triển
- Kết quả đạt được.
- Hạn chế hiện tại.
- Hướng phát triển:
  - Bổ sung dữ liệu đa môi trường.
  - Dùng tracking để gán PPE theo từng người ổn định hơn.
  - Tối ưu mô hình bằng ONNX/TensorRT/OpenVINO.
  - Tích hợp camera IP và hệ thống cảnh báo.
  - Mở rộng class PPE theo yêu cầu thực tế.
