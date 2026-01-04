# Báo Cáo LaTeX - Ứng Dụng Thị Giác Máy Tính Trong Phân Loại Chất Lượng Chuối

## Cấu trúc thư mục

```
report/
├── main.tex                    # File LaTeX chính
├── chapters/
│   ├── cover.tex               # Trang bìa
│   ├── acknowledgement.tex     # Lời cảm ơn
│   ├── chapter1_introduction.tex    # Chương 1: Giới thiệu
│   ├── chapter2_theory.tex          # Chương 2: Cơ sở lý thuyết
│   ├── chapter3_methodology.tex     # Chương 3: Phương pháp thực hiện
│   ├── chapter4_implementation.tex  # Chương 4: Cài đặt và triển khai
│   ├── chapter5_results.tex         # Chương 5: Kết quả và đánh giá
│   ├── chapter6_conclusion.tex      # Chương 6: Kết luận
│   ├── references.tex               # Tài liệu tham khảo
│   └── appendix.tex                 # Phụ lục
├── images/                     # Thư mục chứa hình ảnh
│   └── (thêm hình ảnh vào đây)
└── README.md                   # File hướng dẫn này
```

## Yêu cầu

### Phần mềm cần cài đặt

1. **TeX Distribution**: 
   - Windows: [MiKTeX](https://miktex.org/) hoặc [TeX Live](https://tug.org/texlive/)
   - macOS: [MacTeX](https://tug.org/mactex/)
   - Linux: `sudo apt install texlive-full` (Ubuntu/Debian)

2. **Editor** (khuyến nghị):
   - [TeXstudio](https://www.texstudio.org/)
   - [Overleaf](https://www.overleaf.com/) (online)
   - VS Code + LaTeX Workshop extension

### Font

Báo cáo sử dụng font **Times New Roman**. Đảm bảo font này đã được cài đặt trên hệ thống.

## Cách biên dịch

### Sử dụng XeLaTeX (khuyến nghị)

```bash
cd report
xelatex main.tex
makeglossaries main
xelatex main.tex
xelatex main.tex
```

### Sử dụng TeXstudio

1. Mở file `main.tex` trong TeXstudio
2. Chọn **Options** → **Configure TeXstudio** → **Build**
3. Đặt **Default Compiler** thành `XeLaTeX`
4. Nhấn **F5** hoặc **Build & View**

### Sử dụng Overleaf

1. Upload toàn bộ thư mục `report/` lên Overleaf
2. Chọn **Menu** → **Compiler** → **XeLaTeX**
3. Nhấn **Recompile**

## Thêm hình ảnh

### Logo trường HUS

1. Tải logo Trường Đại học Khoa học Tự nhiên (HUS) 
2. Lưu vào `images/hus_logo.png`
3. Uncomment dòng trong `chapters/cover.tex`:
   ```latex
   \includegraphics[width=0.2\textwidth]{images/hus_logo.png}
   ```

### Ảnh bìa minh họa

1. Chụp screenshot hệ thống hoặc tạo ảnh minh họa
2. Lưu vào `images/cover_image.png`
3. Uncomment dòng tương ứng trong `chapters/cover.tex`

### Ảnh trong nội dung

1. Thêm ảnh vào thư mục `images/`
2. Sử dụng trong LaTeX:
   ```latex
   \begin{figure}[H]
       \centering
       \includegraphics[width=0.8\textwidth]{images/ten_anh.png}
       \caption{Mô tả ảnh}
       \label{fig:ten_label}
   \end{figure}
   ```

## Nội dung báo cáo

### Chương 1: Giới thiệu
- Đặt vấn đề và bối cảnh nghiên cứu
- Mục tiêu nghiên cứu
- Phạm vi nghiên cứu
- Ý nghĩa khoa học và thực tiễn

### Chương 2: Cơ sở lý thuyết
- Tổng quan về Thị giác máy tính
- Mạng nơ-ron tích chập (CNN)
- Kiến trúc YOLO
- Không gian màu và phân tích màu sắc
- Các kỹ thuật xử lý ảnh

### Chương 3: Phương pháp thực hiện
- Kiến trúc hệ thống 2 giai đoạn
- Module BananaAnalyzer
- Phân loại 4 cấp độ
- Feature Refinement

### Chương 4: Cài đặt và triển khai
- Môi trường phát triển
- Huấn luyện mô hình
- Phát triển giao diện người dùng
- Triển khai và đóng gói

### Chương 5: Kết quả và đánh giá
- Kết quả huấn luyện mô hình
- Đánh giá hệ thống thời gian thực
- So sánh với các phương pháp khác
- Hạn chế và thách thức

### Chương 6: Kết luận
- Tổng kết công việc
- Bài học kinh nghiệm
- Hướng phát triển

## Thông tin nhóm

- **Lường Văn Tâm** - MSV: 22001349
- **Khương Thanh Tín** - MSV: 22001349

**Giảng viên hướng dẫn:**
- PGS. TS. Phạm Tiến Lâm
- ThS. Vi Anh Quân

## Lưu ý

1. Báo cáo sử dụng XeLaTeX để hỗ trợ font Unicode tiếng Việt
2. Cần biên dịch 2-3 lần để các cross-reference được cập nhật đầy đủ
3. Nếu gặp lỗi glossary, chạy `makeglossaries main` giữa các lần biên dịch
