# Banana Quality Inspection System (Vietnamese UI)

## 1) Cài đặt
```bash
python -m venv .venv
\.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu PowerShell bị chặn chạy script (ExecutionPolicy), bạn vẫn có thể dùng venv bằng cách gọi thẳng:
```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Font tiếng Việt (quan trọng)
OpenCV `cv2.putText` không hiển thị Unicode tiếng Việt. Ứng dụng này vẽ chữ lên frame bằng Pillow (`ImageDraw`) + file font `.ttf`.

### Gợi ý đường dẫn font trên Windows
- Arial: `C:/Windows/Fonts/arial.ttf`
- Segoe UI: `C:/Windows/Fonts/segoeui.ttf`

Bạn có thể:
- (Khuyến nghị) copy một font hỗ trợ tiếng Việt (vd: Roboto, NotoSans) vào `assets/fonts/Roboto-Regular.ttf`
- Hoặc chỉnh biến `VI_FONT_PATH` trong file `main.py`

Nếu không tìm thấy font, app sẽ **fallback sang English** và in cảnh báo.

## 3) Chạy app
```bash
python main.py
```

## 3.1) (Tuỳ chọn nhưng khuyến nghị) Train YOLOv8
Ứng dụng dùng YOLOv8 (ultralytics). Bạn cần file weights `best.pt` để inference.

### Nếu bạn dùng dataset *Kaggle classification* (folder per class)
Dataset Kaggle thường là **image classification** (không có bounding box). Project này xử lý bằng cách:
- **YOLO detector (COCO)** tìm vị trí quả chuối (bbox) trên webcam.
- **YOLO classifier** (train từ Kaggle) phân loại độ chín trên crop.

Bạn có thể chạy theo 2 mức:
- Mức 1 (nhanh nhất): train **classifier** (Kaggle), dùng detector mặc định `yolov8n.pt`.
- Mức 2 (đúng “train cả 2”): train thêm **detector** riêng (bbox) và đưa vào `weights/detector.pt`.

Script train classifier từ Kaggle: `training_kaggle_classification.py`.

### Train detector (YOLO detection, có bbox)
Bạn cần dataset detection theo chuẩn YOLO (có bbox labels + `data.yaml`).

Chạy (auto dùng GPU nếu có):
```bash
.\.venv\Scripts\python.exe training_script.py --device auto --epochs 100 --imgsz 640 --batch -1
```

Copy weights detector:
- `runs_banana/yolov8n_banana/weights/best.pt` -> `weights/detector.pt`

### Train classifier từ Kaggle (auto-download)
1) Cấu hình Kaggle API (bắt buộc để tải tự động):
- Tạo `kaggle.json` và đặt tại: `%USERPROFILE%/.kaggle/kaggle.json`
	hoặc set env: `KAGGLE_USERNAME`, `KAGGLE_KEY`.
2) Chạy:
```bash
.\.venv\Scripts\python.exe training_kaggle_classification.py --device auto --epochs 80 --imgsz 416 --batch -1
```
3) Copy weights:
- `runs_banana/yolov8n_banana_cls/weights/best.pt` -> `weights/best.pt`

## 3.2) Chọn detector/classifier weights (tuỳ chọn)
Mặc định app sẽ tìm theo thứ tự:
- Classifier: `weights/best.pt` (hoặc đường dẫn trong `main.py`)
- Detector: `weights/detector.pt` -> `runs_banana/yolov8n_banana/weights/best.pt` -> `yolov8n.pt`

Bạn có thể override bằng env:
- `BANANA_DEVICE=cpu` hoặc `BANANA_DEVICE=0`
- `BANANA_DETECTOR_PATH=...` (trỏ tới detector .pt)

Sau khi train xong, weights thường nằm ở:
- `runs_banana/yolov8n_banana/weights/best.pt`

Khuyến nghị copy về:
- `weights/best.pt`

App sẽ tự dò `weights/best.pt` trước; nếu không thấy sẽ hiện trạng thái “Chưa có model YOLO (best.pt)”.

### Mapping class (quan trọng)
- App sẽ cố gắng đọc `datasets/data.yaml` và suy luận mapping từ tên class (vd: `fresh`, `ripe`, `overripe`, `rotten`).
- Nếu dataset của bạn đặt tên khác/lệch thứ tự, hãy chỉnh tay trong [app/grader.py](app/grader.py) biến `class_id_to_category_key`.

## 4) Lưu ý
- Hệ thống hiện dùng YOLOv8 object detection. Độ chính xác phụ thuộc dataset + chất lượng label + ánh sáng khi quay.
