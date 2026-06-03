# PPE Detection Project - Phát hiện thiết bị bảo hộ lao động bằng YOLO

Project này dùng Python và thư viện `ultralytics` để huấn luyện, đánh giá và demo mô hình YOLO cho bài toán phát hiện người lao động cùng thiết bị bảo hộ lao động trong ảnh/video. Phiên bản này được tối ưu để chạy trên **Google Colab**.

## 1. Mục tiêu

- Phát hiện người lao động và các thiết bị bảo hộ trong ảnh/video.
- Nhận dạng các lớp phổ biến: `helmet`, `safety_vest`, `mask`, `gloves`, `goggles`, `boots`, `no_helmet`, `no_vest`.
- Dễ mở rộng class bằng cách sửa `data/data.yaml` theo dataset thật.
- Hiển thị cảnh báo trực tiếp nếu phát hiện người thiếu thiết bị bảo hộ, ví dụ `NO HELMET`, `NO VEST`.
- Hỗ trợ train, detect ảnh, detect video, webcam và demo Streamlit.

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
│   └── .gitkeep
├── requirements.txt
├── README.md
└── report_outline.md
```

## 3. Cài đặt trên Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
# Nếu upload project zip lên Colab thì giải nén trước, hoặc clone repo của bạn.
cd ppe_detection_project
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
nvidia-smi
```

Nếu không có GPU, đặt `--device cpu`, nhưng train sẽ chậm hơn nhiều.

## 4. Chuẩn bị dataset

Project **không tự bịa dataset**. Bạn cần dùng dataset thật và đưa về format YOLO:

```text
ppe_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Mỗi ảnh `abc.jpg` phải có label tương ứng `abc.txt`. Mỗi dòng label YOLO có dạng:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Trong đó các tọa độ đã được normalize về khoảng `[0, 1]`.

### Gợi ý nguồn dataset thật

Bạn có thể tìm dataset PPE từ các nguồn công khai như Roboflow Universe, Kaggle hoặc dataset nội bộ của trường/doanh nghiệp. Khi dùng dataset, hãy kiểm tra license và citation. Không nên trộn dataset nếu class id chưa được chuẩn hóa.

### Convert annotation sang YOLO format

- Nếu annotation là Pascal VOC XML: đọc `xmin, ymin, xmax, ymax`, chia cho kích thước ảnh để tạo `x_center, y_center, width, height`.
- Nếu annotation là COCO JSON: đọc `bbox = [x, y, w, h]`, đổi sang YOLO normalized.
- Nếu dùng Roboflow: chọn export format `YOLOv8` hoặc `YOLOv11`, tải zip và giải nén vào `/content/ppe_dataset`.

Sau đó sửa `data/data.yaml`:

```yaml
path: /content/ppe_dataset
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: helmet
  2: safety_vest
  3: mask
  4: gloves
  5: goggles
  6: boots
  7: no_helmet
  8: no_vest
```

Nếu dataset có class khác, hãy sửa `names` đúng theo thứ tự class id trong label.

## 5. Train model

Train YOLOv8n:

```bash
cd /content/ppe_detection_project
python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```

Train YOLO11n nếu ultralytics hỗ trợ trong môi trường của bạn:

```bash
python src/train.py --data data/data.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --batch 16 --device 0
```

File tốt nhất thường được lưu tại:

```text
runs/train/ppe_yolo/weights/best.pt
```

## 6. Detect ảnh đơn

```bash
python src/detect_image.py \
  --model runs/train/ppe_yolo/weights/best.pt \
  --source /content/test_image.jpg \
  --output runs/detect/image \
  --conf 0.25 \
  --imgsz 640
```

Script sẽ:

- In danh sách object phát hiện được.
- Lưu ảnh có bounding box, label và cảnh báo vào `runs/detect/image`.

## 7. Detect video

```bash
python src/detect_video.py \
  --model runs/train/ppe_yolo/weights/best.pt \
  --source /content/test_video.mp4 \
  --output runs/detect/video \
  --conf 0.25 \
  --imgsz 640
```

Script sẽ xử lý từng frame, vẽ bbox, label, cảnh báo và FPS, sau đó lưu video `.mp4`.

## 8. Webcam real-time

Google Colab không hỗ trợ webcam OpenCV `cv2.VideoCapture(0)` trực tiếp như máy local. File `detect_webcam.py` vẫn có sẵn để dùng khi runtime có webcam forwarding hoặc khi bạn chạy local:

```bash
python src/detect_webcam.py --model runs/train/ppe_yolo/weights/best.pt --camera 0 --conf 0.25
```

Nhấn `q` để thoát.

Nếu chỉ dùng Colab, nên demo real-time bằng cách upload video ngắn hoặc viết thêm cell JavaScript capture webcam rồi gửi frame vào model.

## 9. Chạy Streamlit app

Trên Colab, Streamlit cần tunnel như `localtunnel` hoặc `ngrok`:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Mở URL tunnel, nhập đường dẫn `best.pt`, upload ảnh và xem kết quả.

## 10. Đánh giá model

Đánh giá trên validation set:

```bash
python src/evaluate.py \
  --model runs/train/ppe_yolo/weights/best.pt \
  --data data/data.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Đánh giá trên test set:

```bash
python src/evaluate.py --model runs/train/ppe_yolo/weights/best.pt --data data/data.yaml --split test --device 0
```

Kết quả gồm Precision, Recall, F1-score, mAP50, mAP50-95. Các biểu đồ như confusion matrix sẽ được lưu trong `runs/evaluate` nếu Ultralytics tạo được.

## 11. Ý nghĩa metric

- **Precision**: Tỷ lệ dự đoán dương tính đúng trên tổng số dự đoán dương tính. Precision cao nghĩa là model ít báo nhầm.
- **Recall**: Tỷ lệ object thật được phát hiện đúng. Recall cao nghĩa là model ít bỏ sót PPE hoặc vi phạm.
- **F1-score**: Trung bình điều hòa giữa Precision và Recall. Chỉ số này hữu ích khi cần cân bằng giữa báo nhầm và bỏ sót.
- **mAP50**: Mean Average Precision tại IoU threshold 0.50. Đây là metric phổ biến để đánh giá object detection.
- **mAP50-95**: mAP trung bình trên nhiều ngưỡng IoU từ 0.50 đến 0.95. Chỉ số này khắt khe hơn vì yêu cầu bbox chính xác hơn.

## 12. Logic cảnh báo PPE

File `src/violation_check.py` tách riêng logic kiểm tra vi phạm. Mặc định:

- Nếu có class `person`, bbox người được chia thành vùng đầu, thân, tay và chân.
- `helmet` cần nằm gần vùng đầu.
- `safety_vest` cần nằm gần vùng thân.
- Nếu phát hiện class phủ định như `no_helmet`, `no_vest`, script cảnh báo ngay cả khi dataset không có class `person`.

Bạn có thể thay đổi `required_ppe`, thêm class mới hoặc chỉnh tỷ lệ vùng kiểm tra trong `_person_regions()`.

## 13. Lỗi tiềm ẩn và cách sửa

- **Không tìm thấy model `best.pt`**: hãy train trước hoặc sửa đúng đường dẫn `--model`.
- **Không tìm thấy dataset**: sửa `path` trong `data/data.yaml` đúng với vị trí dataset trên Colab.
- **Class id sai**: đảm bảo thứ tự `names` trong `data.yaml` khớp với số class trong file `.txt`.
- **CUDA out of memory**: giảm `--batch`, giảm `--imgsz`, hoặc dùng model nhỏ hơn như `yolov8n.pt`.
- **Webcam không chạy trên Colab**: dùng detect video/upload ảnh hoặc triển khai webcam bằng JavaScript trong notebook.
- **OpenCV không hiển thị cửa sổ**: Colab không hỗ trợ `cv2.imshow`, hãy lưu ảnh/video hoặc dùng Streamlit/tunnel.

## 14. Hướng phát triển thêm

- Gán PPE theo từng người chính xác hơn bằng tracking hoặc pose estimation.
- Thêm DeepSORT/ByteTrack để theo dõi vi phạm qua video.
- Deploy bằng FastAPI hoặc Streamlit Cloud.
- Tối ưu model bằng ONNX/TensorRT.
- Thêm thống kê số lần vi phạm theo thời gian và xuất báo cáo CSV.
