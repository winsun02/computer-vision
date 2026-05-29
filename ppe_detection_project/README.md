# PPE Detection Project - Phát hiện thiết bị bảo hộ lao động

Dự án này xây dựng một pipeline Computer Vision hoàn chỉnh để **phát hiện người lao động và thiết bị bảo hộ cá nhân (PPE)** trong ảnh, video và webcam real-time. Project dùng Python và thư viện `ultralytics` để train/inference YOLOv8 hoặc YOLOv11.

## 1. Dataset đã chọn

Project này chọn dataset thật, công khai trên Kaggle:

- **Tên dataset**: Construction Site Safety Image Dataset Roboflow
- **Nguồn**: <https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow>
- **Lý do phù hợp**:
  - Có ngữ cảnh công trường/xây dựng đúng với bài toán PPE.
  - Có các nhãn quan trọng cho cảnh báo an toàn: `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Vest`.
  - Có thêm class môi trường như `Safety Cone`, `machinery`, `vehicle`, giúp model học tốt hơn trong scene thực tế.
  - Dataset đã có export kiểu YOLO/Roboflow nên dễ chuẩn hóa về cấu trúc `images/train`, `labels/train`, ...

> Lưu ý: theo yêu cầu hiện tại, project chỉ xử lý các class có thật trong dataset đã chọn; không thêm rule/class ngoài dataset.

## 2. Mục tiêu

- Phát hiện người lao động và các thiết bị bảo hộ.
- Nhận dạng/cảnh báo các lớp từ dataset đã chọn:
  - `person`
  - `helmet` (chuẩn hóa từ `Hardhat`)
  - `safety_vest` (chuẩn hóa từ `Safety Vest`)
  - `mask`
  - `no_helmet` (chuẩn hóa từ `NO-Hardhat`)
  - `no_mask`
  - `no_vest` (chuẩn hóa từ `NO-Safety Vest`)
  - `safety_cone`, `machinery`, `vehicle`
- Chỉ xử lý các class có trong dataset đã chọn; không thêm class/rule ngoài dataset.
- Hiển thị cảnh báo trực tiếp nếu người trong ảnh/video thiếu thiết bị bắt buộc, mặc định là helmet và safety vest.
- Hỗ trợ Windows/Linux, ảnh đơn, video file, webcam và giao diện Streamlit.

## 3. Cấu trúc project

```text
ppe_detection_project/
├── data/
│   ├── data.yaml
│   └── css_construction_safety.yaml
├── src/
│   ├── prepare_css_dataset.py
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

## 4. Cài đặt môi trường

### 4.1. Tạo virtual environment

Windows PowerShell:

```powershell
cd ppe_detection_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
cd ppe_detection_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra cài đặt:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

Nếu dùng GPU NVIDIA, hãy cài PyTorch bản CUDA phù hợp theo hướng dẫn chính thức của PyTorch.

## 5. Chuẩn bị dataset đã chọn

### 5.1. Cách 1 - tải thủ công từ Kaggle

1. Mở trang dataset: <https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow>
2. Tải file `.zip` về máy.
3. Chạy script chuẩn hóa dataset:

```bash
python src/prepare_css_dataset.py --source path/to/archive.zip --output datasets/construction_site_safety --yaml-output data/data.yaml --copy --force
```

Nếu bạn đã giải nén dataset:

```bash
python src/prepare_css_dataset.py --source path/to/extracted_folder --output datasets/construction_site_safety --yaml-output data/data.yaml --copy --force
```

### 5.2. Cách 2 - tải bằng Kaggle CLI

Cần có file credential `~/.kaggle/kaggle.json` hoặc biến môi trường Kaggle hợp lệ.

```bash
python src/prepare_css_dataset.py --download --output datasets/construction_site_safety --yaml-output data/data.yaml --copy --force
```

Script `prepare_css_dataset.py` sẽ:

- tìm các split YOLO từ dataset gốc như `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`;
- copy hoặc symlink ảnh sang `datasets/construction_site_safety/images/<split>`;
- rewrite nhãn YOLO sang class ID canonical của project;
- đổi `valid` thành `val`;
- ghi lại `data/data.yaml` và `dataset_metadata.json`.

### 5.3. Cấu trúc sau khi chuẩn hóa

```text
datasets/construction_site_safety/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset_metadata.json
```

### 5.4. Class mapping của dataset đã chọn

| Class gốc | Class chuẩn hóa |
|---|---|
| `Hardhat` | `helmet` |
| `Mask` | `mask` |
| `NO-Hardhat` | `no_helmet` |
| `NO-Mask` | `no_mask` |
| `NO-Safety Vest` | `no_vest` |
| `Person` | `person` |
| `Safety Cone` | `safety_cone` |
| `Safety Vest` | `safety_vest` |
| `machinery` | `machinery` |
| `vehicle` | `vehicle` |

## 6. File `data/data.yaml`

`data/data.yaml` hiện trỏ đến dataset đã chuẩn hóa:

```yaml
path: ../datasets/construction_site_safety
train: images/train
val: images/val
test: images/test
names:
  0: helmet
  1: mask
  2: no_helmet
  3: no_mask
  4: no_vest
  5: person
  6: safety_cone
  7: safety_vest
  8: machinery
  9: vehicle
```

Nếu bạn dùng dataset PPE khác, chỉ cần sửa `names` và đường dẫn cho khớp với label YOLO của dataset đó.

## 7. Train model

Ví dụ train YOLOv8 nano:

```bash
python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```

Ví dụ train YOLO11 nano:

```bash
python src/train.py --data data/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```

Nếu không có GPU:

```bash
python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 20 --imgsz 640 --batch 8 --device cpu
```

Sau khi train, model tốt nhất thường nằm tại:

```text
runs/train/ppe_yolo/weights/best.pt
```

## 8. Detect ảnh đơn

```bash
python src/detect_image.py --model runs/train/ppe_yolo/weights/best.pt --source path/to/image.jpg --output runs/detect/image --conf 0.25
```

Kết quả:

- Ảnh output có bounding box, label và cảnh báo được lưu trong `runs/detect/image`.
- Terminal in danh sách object phát hiện được.

## 9. Detect video

```bash
python src/detect_video.py --model runs/train/ppe_yolo/weights/best.pt --source path/to/video.mp4 --output runs/detect/video --conf 0.25
```

Hiển thị preview khi chạy:

```bash
python src/detect_video.py --model runs/train/ppe_yolo/weights/best.pt --source path/to/video.mp4 --show
```

Kết quả video được lưu ở `runs/detect/video/*_detected.mp4`. FPS được vẽ trực tiếp trên từng frame.

## 10. Chạy webcam real-time

```bash
python src/detect_webcam.py --model runs/train/ppe_yolo/weights/best.pt --camera 0 --conf 0.25
```

Nhấn `q` để thoát.

## 11. Chạy Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Sau đó mở URL Streamlit trên trình duyệt, nhập đường dẫn model, upload ảnh và xem kết quả.

## 12. Đánh giá model

Đánh giá trên validation set:

```bash
python src/evaluate.py --model runs/train/ppe_yolo/weights/best.pt --data data/data.yaml --split val
```

Đánh giá trên test set:

```bash
python src/evaluate.py --model runs/train/ppe_yolo/weights/best.pt --data data/data.yaml --split test
```

Script sẽ in:

- Precision
- Recall
- F1-score
- mAP50
- mAP50-95

Nếu Ultralytics sinh được confusion matrix, file sẽ nằm trong thư mục `runs/evaluate/ppe_eval`.

## 13. Ý nghĩa các metric

- **Precision**: trong các object model dự đoán là đúng, tỷ lệ bao nhiêu thực sự đúng. Precision cao nghĩa là ít false positive.
- **Recall**: trong các object thật sự tồn tại, model tìm ra được bao nhiêu. Recall cao nghĩa là ít false negative.
- **F1-score**: trung bình điều hòa giữa Precision và Recall. Metric này hữu ích khi cần cân bằng giữa phát hiện thiếu và cảnh báo nhầm.
- **IoU (Intersection over Union)**: mức độ chồng lấp giữa bounding box dự đoán và bounding box ground truth.
- **AP (Average Precision)**: diện tích dưới đường Precision-Recall cho một class.
- **mAP50**: mean Average Precision tại ngưỡng IoU = 0.50.
- **mAP50-95**: mAP trung bình trên nhiều ngưỡng IoU từ 0.50 đến 0.95. Đây là metric nghiêm ngặt hơn mAP50.

## 14. Logic cảnh báo vi phạm PPE

File `src/violation_check.py` tách riêng rule kiểm tra vi phạm:

- Nếu có class `person`, mỗi người được kiểm tra độc lập.
- `helmet` được tìm trong vùng đầu.
- `safety_vest` được tìm trong vùng thân.
- `mask` được tìm trong vùng đầu nếu bạn thêm `mask` vào danh sách PPE bắt buộc.
- Nếu dataset không có class `person`, hệ thống vẫn xử lý được bằng cách:
  - báo cảnh báo khi phát hiện class âm như `no_helmet`, `no_vest`, `no_mask`;
  - cảnh báo global nếu có PPE object nhưng thiếu class bắt buộc.

Muốn thay đổi PPE bắt buộc, sửa `DEFAULT_REQUIRED_PPE` trong `src/violation_check.py` hoặc truyền danh sách khác khi gọi function `check_ppe_violations`.

## 15. Lỗi tiềm ẩn và cách sửa

### Kaggle CLI không tải được dataset

Nguyên nhân thường gặp:

- chưa cài `kaggle`;
- chưa cấu hình `~/.kaggle/kaggle.json`;
- chưa accept điều khoản dataset trên Kaggle;
- mạng bị chặn.

Cách sửa: tải dataset thủ công từ Kaggle rồi dùng `--source path/to/archive.zip`.

### Không tìm thấy model `best.pt`

Nguyên nhân: chưa train hoặc sai đường dẫn.

Cách sửa:

```bash
python src/train.py --data data/data.yaml --model yolov8n.pt
```

Sau đó dùng đúng path `runs/train/ppe_yolo/weights/best.pt`.

### Không tìm thấy dataset

Nguyên nhân: chưa chạy `src/prepare_css_dataset.py` hoặc `path` trong `data/data.yaml` chưa đúng.

Cách sửa: chuẩn hóa dataset theo mục 5 hoặc sửa `path`, `train`, `val`, `test` trong YAML để trỏ đúng folder dataset.

### Webcam không mở được

Nguyên nhân: sai camera index hoặc môi trường server không có camera.

Cách sửa:

```bash
python src/detect_webcam.py --camera 1
```

hoặc chạy trên máy local có webcam.

### Video output không mở được

Nguyên nhân: codec MP4 trên hệ điều hành thiếu hoặc OpenCV không hỗ trợ.

Cách sửa: thử đổi extension/codec trong `src/detect_video.py`, ví dụ dùng `XVID` và `.avi`.

### Cảnh báo thiếu PPE chưa chính xác

Nguyên nhân: rule hiện tại dùng heuristic theo vùng bounding box. Với công trường đông người hoặc vật thể bị che khuất, việc gán PPE vào từng người có thể sai.

Cách cải thiện:

- Dùng tracking để giữ ID người qua nhiều frame.
- Tinh chỉnh vùng `head`, `torso` trong `src/utils.py`.
- Dùng pose estimation để xác định vùng cơ thể chính xác hơn.
- Nếu đổi sang dataset khác, chỉ thêm class/rule sau khi class đó thật sự tồn tại trong dataset mới.

## 16. Hướng phát triển thêm

- Thêm tracking ByteTrack/DeepSORT để theo dõi từng công nhân.
- Thêm rule theo khu vực nguy hiểm trong frame.
- Export model sang ONNX/TensorRT/OpenVINO để chạy edge device.
- Tạo dashboard thống kê số lần vi phạm theo thời gian.
- Tích hợp camera IP/RTSP.
- Tối ưu threshold riêng cho từng class.
