# Kế hoạch nghiên cứu: Detect chuối xanh/chín + nải chuối + chuối hỏng (tự huấn luyện, không dùng model mua sẵn qua API)

Mục tiêu: từ ảnh/video (webcam/điện thoại), hệ thống phải:
1) Nhận biết chuối **xanh** vs **chín** (có thể thêm quá chín).
2) Xử lý **nải/buồng** (nhiều quả trong cùng khung hình) và quyết định “có quả nào hỏng không”.
3) Phát hiện **chuối hỏng** (mốc/thối/đen bất thường) đủ tin cậy để loại bỏ.

Ràng buộc: không dùng dịch vụ “model có sẵn” kiểu gọi API (Roboflow API). Thư viện open-source dùng thoải mái.

---

## 1) Trạng thái hiện tại của repo (để tận dụng)
Repo hiện đã có:
- Pipeline **2-stage**: YOLO detector tìm bbox chuối → YOLO classifier phân loại độ chín trên crop (xem [app/grader.py](../app/grader.py)).
- Module **BananaAnalyzer** có phân tích HSV/LAB + đếm đốm + texture/morphology (xem [app/banana_analyzer.py](../app/banana_analyzer.py)).

Điểm còn thiếu cho bài toán “nải chuối”:
- App hiện thiên về **1 bbox chính**/frame. Muốn “bóc nải” cần xử lý **nhiều bbox/instance** và hợp nhất kết quả.

---

## 2) Định nghĩa bài toán (đúng ngay từ đầu)
Bạn nên chốt rõ 3 lớp quyết định, vì label sai/không thống nhất là nguyên nhân số 1 làm model kém.

### 2.1. Cấp độ quả (instance-level)
Đề xuất nhãn tối thiểu (phù hợp repo):
- `unripe` (xanh)
- `export` (chín vừa/vàng đẹp)
- `overripe` (quá chín/nhiều đốm)
- `defective` (hỏng: thối/mốc/đen bất thường/dập nặng)

### 2.2. Cấp độ nải (group-level)
Một nải được xem là **defective** nếu *bất kỳ* quả nào `defective`.
Tuỳ yêu cầu business, bạn có thể đặt rule:
- Nếu có >= 30% quả `overripe` → nải “cần bán gấp”.

### 2.3. “Hỏng” là gì?
Cực quan trọng: hỏng do **mốc/thối** khác với **đốm chín**. Nếu bạn gom hết vào 1 class, model sẽ rất hay nhầm.
Gợi ý: trong nhãn `defective` chỉ gồm các trường hợp thật sự cần loại bỏ.

---

## 3) Kiến trúc giải pháp (khuyến nghị)
Bạn có 2 hướng. Hướng A dễ hơn và phù hợp repo.

### Hướng A (khuyến nghị): Detector nhiều quả + classifier/feature refine
1) **Detector**: phát hiện *tất cả* quả chuối trong frame (multi-bbox).
   - Output: danh sách bbox quả chuối.
2) Với mỗi bbox:
   - **Classifier** (YOLO classification hoặc CNN): dự đoán `unripe/export/overripe/defective`.
   - **Feature refinement** (BananaAnalyzer): dùng HSV ratio + black/brown ratio + texture/spot_count để “đè” quyết định khi có tín hiệu hỏng mạnh.
3) **Aggregation cho nải**:
   - Nếu có bất kỳ quả `defective` → nải defective.
   - Nếu không, lấy mode/average theo confidence.

Ưu: label rẻ, code tận dụng sẵn, chạy realtime ổn.
Nhược: “hỏng” dạng tinh vi (mốc nhẹ) có thể cần thêm data/feature.

### Hướng B (nâng cao): Instance segmentation trên nải + phân loại từng quả
- Dùng YOLO segmentation hoặc Mask R-CNN để tách từng quả ngay cả khi chồng lấp.
- Sau đó phân loại/đánh giá từng mask.

Ưu: tách quả trong nải tốt hơn.
Nhược: label segmentation tốn công hơn bbox.

---

## 4) Data: cách thu thập + chuẩn hoá để đạt mục tiêu
### 4.1. Thu thập tối thiểu (để bắt đầu ra kết quả)
- Mỗi class tối thiểu 500–1500 ảnh (càng đa dạng càng tốt).
- Chụp đa dạng:
  - Nền khác nhau (bếp, chợ, bàn gỗ, sàn)
  - Ánh sáng khác nhau (đèn vàng, daylight, ngoài trời)
  - Góc chụp (top-down, ngang, cận)
  - Có và không có nải

Nếu thiếu data `defective`, bạn sẽ thấy model “ngại” dự đoán defective.

### 4.2. Chiến lược label rẻ nhất cho nải
- Label bbox quả chuối (detection) + nhãn class cho từng bbox.
- Không cần segmentation ngay.
- Với nải: không cần label riêng, vì nải = aggregation từ quả.

### 4.3. Tool label miễn phí / không phụ thuộc API
- LabelImg (desktop)
- CVAT self-host (nếu team)
- makesense.ai (web) xuất YOLO

---

## 5) Training: đề xuất cụ thể (không dùng model “mua sẵn”)
### 5.1. Làm rõ “model có sẵn” nghĩa là gì
Có 2 mức hiểu:
1) Không dùng dịch vụ/API trả phí (Roboflow hosted inference): **OK**, bạn vẫn có thể dùng pretrained open-source (COCO/ImageNet) để fine-tune.
2) Không dùng cả pretrained weights: bạn phải train from-scratch → cần data lớn hơn nhiều (thường khó đạt nhanh).

Theo thông tin bạn chốt: **được phép dùng pretrained open-source** → hướng (1) là đúng và thực tế nhất.

### 5.2. Detector
- Train YOLO detection với 1 class `banana` hoặc 4 class theo độ chín/hỏng.
  - Khuyến nghị: **detector chỉ cần `banana`** trước (để ổn định bbox), còn phân loại để classifier xử lý.
- Pretrained: dùng `yolov8n.pt` làm khởi tạo (open-source).

### 5.3. Classifier
- Train YOLO classifier từ dataset folder-per-class (repo đã có script Kaggle).
- Bạn nên fine-tune trên **ảnh crop thực tế** từ camera của bạn (domain adaptation). Cách nhanh:
  - Chạy detector trên video thực tế
  - Cắt crop bbox ra folder theo class (tự gán nhãn bằng tay vài nghìn crop)
  - Train classifier lại

### 5.4. Feature-based refinement cho “defective”
Tận dụng [app/banana_analyzer.py](../app/banana_analyzer.py):
- `black_ratio` cao + texture variance cao + spot_count cao → tăng xác suất defective.
- Mục tiêu: giảm false negative (bỏ sót hỏng).

---

## 6) Đánh giá (evaluation) đúng với nhu cầu
### 6.1. Theo quả
- Confusion matrix cho 4 class.
- Ưu tiên metric cho `defective`:
  - Recall(defective) cao (ít bỏ sót)
  - Precision(defective) vừa đủ (không loại nhầm quá nhiều)

### 6.2. Theo nải
- Một test set chứa video/nhiều quả.
- Rule đánh giá: nải defective nếu có ≥1 quả defective.
- Metric: recall/precision của nải defective.

---

## 7) Gợi ý thay đổi trong app (khi bạn muốn triển khai)
Repo hiện phù hợp 1 bbox/khung. Để “bóc nải”, cần:
- Detector trả về **N bbox**.
- Với mỗi bbox: classifier + analyzer → GradeResult.
- UI overlay nhiều bbox + danh sách kết quả.
- Rule tổng hợp: “nải có hỏng không”.

Repo đã có thể nâng cấp theo hướng này (multi-bbox) để **khoanh đúng “quả nào hỏng”**. Khi chạy, bạn có thể giới hạn số quả xử lý mỗi frame bằng biến môi trường:
- `BANANA_MAX_FRUITS=6` (mặc định) để giữ realtime.

---

## 8) Câu hỏi cần bạn chốt (để nghiên cứu đúng hướng)
1) Bạn có cho phép dùng **pretrained open-source** (vd `yolov8n.pt`, `yolov8n-cls.pt`) để fine-tune không, hay bắt buộc train from scratch?
2) Bạn muốn output ở mức nào: chỉ “nải có hỏng/không”, hay phải highlight **quả nào hỏng**?
3) Camera/điều kiện chạy: webcam cố định trong xưởng (ánh sáng ổn định) hay dùng điện thoại ngoài chợ (ánh sáng biến thiên mạnh)?

Bạn đã chốt:
- Cho phép pretrained open-source.
- Bắt buộc highlight đúng quả hỏng.
- Ưu tiên kịch bản: **cầm quả chuối trên tay** (ánh sáng vừa đủ, không cần quá khắc nghiệt).

---

## 9) Hướng dẫn data cho kịch bản “cầm chuối trên tay” (để model ổn định)
Mục tiêu của phần này là làm cho detector/classifier “miễn nhiễm” với nền + da tay + phản xạ ánh sáng.

Checklist khi quay/chụp data:
- Khoảng cách: giữ quả chiếm ~20–60% khung hình (đừng quá nhỏ).
- Nền: 50% ảnh nền đơn giản, 50% ảnh nền lộn xộn (chợ/quầy hàng) để model không overfit.
- Tay cầm: đổi tay trái/phải, da tay khác nhau, có/không có găng.
- Ánh sáng: trong nhà (đèn vàng), ngoài trời râm, ngoài trời nắng nhẹ; tránh cháy sáng mạnh.
- Motion blur: chủ động chụp một ít ảnh hơi rung để mô phỏng thực tế.

Labeling tối thiểu để highlight quả hỏng:
- Với mỗi ảnh nhiều quả/nải: vẽ bbox **từng quả** và gán nhãn `unripe/export/overripe/defective`.
- Đặc biệt với `defective`: tách rõ “đốm chín” vs “mốc/thối/đen bất thường/dập nặng”.

Đề xuất target metric cho `defective`:
- Ưu tiên **recall cao** (ít bỏ sót quả hỏng), chấp nhận precision giảm nhẹ ở giai đoạn đầu.
