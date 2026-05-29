# PPE Detection Project

Project Python dùng YOLO qua thư viện Ultralytics để **phát hiện và nhận dạng thiết bị bảo hộ lao động (PPE)** trong ảnh, video và webcam real-time. Project phù hợp cho bài tập lớn học thuật và có thể mở rộng thành demo thực tế.

## 1. Mục tiêu

- Phát hiện người lao động và các thiết bị bảo hộ trong ảnh/video.
- Nhận dạng các lớp PPE phổ biến:
  - `helmet` / `hardhat`
  - `safety vest`
  - `mask`
  - `gloves`
  - `goggles`
  - `boots`
  - `no-helmet`
  - `no-vest`
- Cho phép mở rộng số lượng class bằng cách sửa `data/data.yaml`.
- Hiển thị cảnh báo trực tiếp nếu phát hiện người không đủ PPE, ví dụ `NO HELMET`, `NO VEST`.
- Hỗ trợ 3 chế độ inference:
  1. Detect ảnh đơn.
  2. Detect video.
  3. Detect webcam real-time.

## 2. Cấu trúc project

```text
ppe_detection_project/
├── data/
│   └── data.yaml
├── src/
│   ├── train.py
│   ├── detect_image.py
│   ├── detect_video.py
│   ├── detect_webcam.py
│   ├── evaluate.py
│   ├── violation_check.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── runs/
├── requirements.txt
├── README.md
└── report_outline.md
```

## 3. Cài đặt

### 3.1. Tạo môi trường ảo

Windows PowerShell:

```powershell
cd ppe_detection_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
cd ppe_detection_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2. Kiểm tra GPU

Nếu dùng NVIDIA GPU, cài PyTorch bản CUDA phù hợp theo hướng dẫn chính thức của PyTorch. Sau đó chạy:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu kết quả là `False`, vẫn có thể chạy bằng CPU với `--device cpu`, nhưng tốc độ sẽ chậm hơn.

## 4. Chuẩn bị dataset

Project **không tự bịa dataset**. Bạn cần tự tải, gán nhãn hoặc chuyển đổi dataset hợp lệ sang YOLO format.

### 4.1. Cấu trúc YOLO bắt buộc

Ví dụ đặt dataset ở `dataset/ppe_yolo` cạnh thư mục project:

```text
dataset/ppe_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Mỗi ảnh phải có file label `.txt` cùng tên trong thư mục `labels/<split>`. Ví dụ:

```text
images/train/0001.jpg
labels/train/0001.txt
```

### 4.2. Định dạng label YOLO

Mỗi dòng trong file `.txt` có dạng:

```text
class_id x_center y_center width height
```

Trong đó `x_center`, `y_center`, `width`, `height` được chuẩn hóa về khoảng `[0, 1]` theo kích thước ảnh.

Ví dụ:

```text
0 0.512 0.480 0.240 0.700
1 0.510 0.160 0.100 0.120
```

### 4.3. Cấu hình `data/data.yaml`

File mẫu đã có sẵn:

```yaml
path: ../dataset/ppe_yolo
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: helmet
  2: safety vest
  3: mask
  4: gloves
  5: goggles
  6: boots
  7: no-helmet
  8: no-vest
```

Bạn phải sửa `path` và `names` cho khớp chính xác với dataset của mình. Thứ tự class trong `names` chính là `class_id` trong label `.txt`.

### 4.4. Gợi ý cách tạo hoặc convert dataset

- Nếu dataset đang ở Pascal VOC XML:
  - Đọc `xmin, ymin, xmax, ymax` và kích thước ảnh.
  - Tính:
    - `x_center = ((xmin + xmax) / 2) / image_width`
    - `y_center = ((ymin + ymax) / 2) / image_height`
    - `width = (xmax - xmin) / image_width`
    - `height = (ymax - ymin) / image_height`
  - Ghi ra file `.txt` theo format YOLO.
- Nếu dataset đang ở COCO JSON:
  - Với mỗi annotation, lấy `category_id` và bbox `[x, y, width, height]`.
  - Map `category_id` sang class index liên tục từ `0..N-1`.
  - Chuẩn hóa bbox theo kích thước ảnh.
- Nếu tự gán nhãn:
  - Có thể dùng CVAT, LabelImg, Roboflow hoặc Label Studio.
  - Chọn export format là YOLO nếu công cụ hỗ trợ.
- Cần kiểm tra thủ công một số ảnh sau khi convert để tránh nhầm class hoặc bbox lệch.

## 5. Train model

Chạy từ thư mục `ppe_detection_project`:

```bash
python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```

Nếu không có GPU:

```bash
python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu
```

Có thể dùng YOLOv11 nếu phiên bản Ultralytics của bạn hỗ trợ checkpoint tương ứng, ví dụ:

```bash
python src/train.py --data data/data.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --batch 16 --device 0
```

Checkpoint tốt nhất thường nằm tại:

```text
runs/train/ppe_yolo/weights/best.pt
```

## 6. Detect ảnh đơn

```bash
python src/detect_image.py --model runs/train/ppe_yolo/weights/best.pt --source path/to/image.jpg --output runs/detect/image --conf 0.25
```

Kết quả:

- Ảnh đã vẽ bounding box, label, confidence.
- Cảnh báo thiếu PPE nếu rule phát hiện vi phạm.
- Danh sách object in ra terminal.

## 7. Detect video

```bash
python src/detect_video.py --model runs/train/ppe_yolo/weights/best.pt --source path/to/video.mp4 --output runs/detect/video --conf 0.25
```

Nếu muốn xem trực tiếp trong lúc xử lý:

```bash
python src/detect_video.py --model runs/train/ppe_yolo/weights/best.pt --source path/to/video.mp4 --show
```

Output video `.mp4` được lưu trong `runs/detect/video` và có hiển thị FPS.

## 8. Detect webcam real-time

```bash
python src/detect_webcam.py --model runs/train/ppe_yolo/weights/best.pt --camera 0 --conf 0.25
```

Nhấn `q` để thoát.

Nếu máy có nhiều webcam, thử `--camera 1`, `--camera 2`, ...

## 9. Chạy Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Trên giao diện web:

1. Nhập đường dẫn model `best.pt`.
2. Upload ảnh.
3. Xem ảnh kết quả, danh sách object và cảnh báo PPE.

## 10. Đánh giá model

Đánh giá trên validation set:

```bash
python src/evaluate.py --model runs/train/ppe_yolo/weights/best.pt --data data/data.yaml --split val --imgsz 640 --batch 16 --device 0
```

Đánh giá trên test set:

```bash
python src/evaluate.py --model runs/train/ppe_yolo/weights/best.pt --data data/data.yaml --split test --imgsz 640 --batch 16 --device 0
```

Script sẽ in:

- Precision
- Recall
- F1-score
- mAP50
- mAP50-95

Với `plots=True`, Ultralytics thường lưu thêm các biểu đồ như confusion matrix trong thư mục `runs/evaluate/ppe_eval`.

## 11. Giải thích metric

### Precision

Precision cho biết trong các object mà model dự đoán là đúng, có bao nhiêu dự đoán thật sự đúng.

```text
Precision = TP / (TP + FP)
```

Precision cao nghĩa là model ít báo nhầm.

### Recall

Recall cho biết trong các object thật sự tồn tại, model tìm được bao nhiêu object.

```text
Recall = TP / (TP + FN)
```

Recall cao nghĩa là model ít bỏ sót object.

### F1-score

F1-score là trung bình điều hòa giữa Precision và Recall.

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

F1 hữu ích khi cần cân bằng giữa báo nhầm và bỏ sót.

### mAP

AP là Average Precision của một class trên nhiều ngưỡng confidence. mAP là trung bình AP trên nhiều class.

- `mAP50`: mAP tại ngưỡng IoU = 0.50.
- `mAP50-95`: mAP trung bình trên các ngưỡng IoU từ 0.50 đến 0.95, thường khắt khe hơn và phản ánh chất lượng bbox tốt hơn.

## 12. Logic cảnh báo PPE

File `src/violation_check.py` chứa logic rule-based, dễ chỉnh sửa:

- Nếu detect class `no-helmet` thì cảnh báo `NO HELMET`.
- Nếu detect class `no-vest` thì cảnh báo `NO VEST`.
- Nếu có class `person`, hệ thống kiểm tra:
  - Helmet/hardhat có overlap với vùng đầu của person không.
  - Safety vest có overlap với vùng thân của person không.
- Nếu dataset không có class `person`, hệ thống vẫn xử lý được bằng cách:
  - Dựa vào class vi phạm trực tiếp như `no-helmet`, `no-vest`.
  - Nếu chỉ có object PPE nhưng thiếu helmet/vest, hiển thị cảnh báo fallback ở mức ảnh.

Trong demo thực tế, nên kết hợp tracking hoặc pose estimation để gán PPE theo từng người chính xác hơn.

## 13. Lỗi tiềm ẩn và cách sửa

### Không tìm thấy model `best.pt`

Nguyên nhân: chưa train hoặc sai đường dẫn.

Cách sửa:

```bash
python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 50
```

Sau đó kiểm tra `runs/train/ppe_yolo/weights/best.pt`.

### Không tìm thấy dataset

Nguyên nhân: `path` trong `data/data.yaml` chưa đúng.

Cách sửa: mở `data/data.yaml`, sửa `path` về đúng thư mục dataset gốc.

### Class dự đoán sai tên

Nguyên nhân: thứ tự `names` trong `data.yaml` không khớp `class_id` trong label.

Cách sửa: kiểm tra lại mapping class và label `.txt`.

### Webcam không mở được

Nguyên nhân: sai index camera hoặc camera đang bị phần mềm khác dùng.

Cách sửa:

```bash
python src/detect_webcam.py --camera 1
```

hoặc đóng ứng dụng đang dùng webcam.

### Video output không mở được

Nguyên nhân: codec OpenCV trên máy không tương thích.

Cách sửa: thử cài lại `opencv-python`, đổi đuôi output hoặc chuyển codec trong `src/utils.py`.

### Chạy CPU quá chậm

Cách sửa:

- Dùng model nhỏ như `yolov8n.pt` hoặc `yolo11n.pt`.
- Giảm `--imgsz` xuống 416 hoặc 320.
- Dùng GPU nếu có.

## 14. Hướng phát triển thêm

- Thêm tracking để theo dõi từng người qua nhiều frame.
- Thêm rule cho `mask`, `gloves`, `goggles`, `boots` theo yêu cầu công trường.
- Tối ưu inference bằng ONNX, TensorRT hoặc OpenVINO.
- Tích hợp camera IP/RTSP.
- Lưu log vi phạm theo thời gian.
- Gửi cảnh báo qua email, Telegram, dashboard hoặc hệ thống quản lý an toàn.
- Mở rộng dataset theo nhiều điều kiện ánh sáng, góc quay và môi trường làm việc.
