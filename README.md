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

### Check nhanh thiếu gì (khuyến nghị khi quay lại project)
```bash
python check_setup.py
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

### Không có Roboflow thì làm detector sao cho rẻ?
Bạn có 3 lựa chọn (từ dễ → khó):
- **Dùng sẵn detector COCO**: cứ để `yolov8n.pt` như mặc định (0 chi phí, thường đủ demo).
- **Tự label bbox bằng tool miễn phí** rồi train YOLO detection: dùng `labelImg` / `CVAT self-host` / `makesense.ai` (web miễn phí) để xuất YOLO format.
- **Haar Cascade (classical CV)**: không cần train YOLO detector, nhưng độ ổn định thường kém khi nền/ánh sáng thay đổi.

Haar Cascade backend (nếu bạn có `haarbanana.xml`):
```powershell
$env:BANANA_DETECTOR_BACKEND = "haar"
$env:BANANA_HAAR_PATH = "haarbanana.xml"
python main.py
```

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

### Tăng ổn định khi chưa có detector custom dataset
Nếu bạn chưa có data bbox để train detector riêng, bạn vẫn có thể tăng ổn định (giảm nhấp nháy bbox) bằng cách giữ bbox gần nhất vài frame khi detector COCO miss tạm thời:

```powershell
$env:BANANA_BBOX_HOLD = "5"   # default=5, set 0 để tắt
python main.py
```

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

## 5) Build Android (APK) có được không?
Có, nhưng **không thể “đóng gói trực tiếp”** project hiện tại thành APK theo kiểu 1-click.

Lý do:
- UI đang dùng `customtkinter` (desktop only).
- Inference đang dùng `ultralytics` (thường kéo theo PyTorch), rất khó/không thực tế để nhét vào Android runtime.

### Hướng khuyến nghị (chạy on-device): Export sang TFLite + Android Studio
1) Export model sang `.tflite`:
```bash
python export_android_models.py --classifier weights/best.pt --detector yolov8n.pt --imgsz 416
```
Kết quả nằm trong `exports_android/`.

2) Tạo app Android (Kotlin) dùng:
- CameraX để lấy frame camera
- TensorFlow Lite để chạy:
	- Detector (lọc bbox class `banana`)
	- Classifier (chấm độ chín trên crop)
- Vẽ bbox + text overlay lên preview

Gợi ý thư viện phía Android:
- `org.tensorflow:tensorflow-lite`
- (tuỳ chọn) `org.tensorflow:tensorflow-lite-support`

### Hướng nhanh nhưng cần mạng: Android client + Python server
Nếu bạn muốn ra app nhanh để demo: Android chỉ gửi frame lên server (PC/cloud) chạy `ultralytics`, nhận kết quả về để overlay.

Nếu bạn muốn mình làm tiếp phần Android:
- Bạn muốn **chạy offline on-device (TFLite)** hay **chạy qua server**?
- Bạn muốn giữ UI giống desktop (1 màn hình camera + panel) hay UI tối giản?

## 6) Build thẳng sang .exe (Windows) có được không?
Có. Bạn có thể đóng gói thành app Windows bằng **PyInstaller**.

Lưu ý quan trọng:
- Vì có `ultralytics/torch`, file build sẽ **rất nặng** và thời gian build lâu (đây là bình thường với PyTorch).
- Khuyến nghị build kiểu **one-folder** (thư mục `dist/...`) để ổn định hơn. One-file vẫn làm được nhưng thường chậm mở và dễ bị antivirus “soi”.

### Cách build nhanh
1) (Khuyến nghị) dùng venv:
```bash
python -m venv .venv
\.\.venv\Scripts\activate
```

2) Build:
```powershell
./build_exe.ps1
```

Output:
- `dist/BananaQualityGrading/BananaQualityGrading.exe`

### Nếu mở exe mà báo thiếu model/font
App cần các file runtime sau (có thể để app tự tải nếu bạn set URL ENV trong `main.py`):
- `weights/best.pt` (classifier)
- `weights/detector.pt` hoặc `yolov8n.pt` (detector)
- `assets/fonts/Roboto-Regular.ttf` (font tiếng Việt)

PyInstaller spec đã cố gắng “nhặt” những thứ này nếu có trong repo: `banana_quality_grading.spec`.
