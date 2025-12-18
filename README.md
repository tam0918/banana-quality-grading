# Banana Quality Inspection System (Vietnamese UI)

## 1) Cài đặt
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
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

1) Chuẩn bị dataset định dạng YOLOv8 (có `data.yaml`).
2) Giải nén dataset vào thư mục `datasets/` sao cho có `datasets/data.yaml`.
3) (Nếu cần) chỉnh đường dẫn dataset trong `training_script.py`.
3) Chạy:
```bash
python training_script.py
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
