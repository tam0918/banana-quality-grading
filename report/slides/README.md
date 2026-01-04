# Slide Báo cáo - Phân loại Chất lượng Chuối

## Cấu trúc thư mục

```
slides/
├── main.tex          # File slide chính (Beamer)
├── README.md         # File này
└── images/           # Ảnh sử dụng trong slide
    ├── results.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── train_batch0.jpg
    ├── train_batch1.jpg
    ├── real_e2e.jpg
    └── pipeline_results/
        ├── 3.jpg
        ├── 5.jpg
        ├── 7.jpg
        ├── 14.jpg
        └── 15.jpg
```

## Biên dịch slide

### Yêu cầu
- LaTeX distribution (TeX Live, MiKTeX, hoặc MacTeX)
- Các package: `beamer`, `tikz`, `pgfplots`, `fontawesome5`, `babel` (Vietnamese)

### Cách biên dịch

**Sử dụng pdflatex:**
```bash
cd report/slides
pdflatex main.tex
pdflatex main.tex  # Chạy 2 lần để cập nhật TOC
```

**Sử dụng latexmk (khuyên dùng):**
```bash
cd report/slides
latexmk -pdf main.tex
```

**Trong VS Code với LaTeX Workshop:**
1. Mở file `main.tex`
2. Nhấn `Ctrl+Alt+B` hoặc `Cmd+Alt+B` (Mac) để build
3. Hoặc nhấn nút "Build LaTeX project" trên toolbar

## Nội dung slide

1. **Giới thiệu** - Đặt vấn đề và mục tiêu
2. **Phương pháp** - Kiến trúc 2-stage, BananaAnalyzer
3. **Dữ liệu & Huấn luyện** - Kaggle dataset, training process
4. **Kết quả** - Giai đoạn 1 (classifier) và Giai đoạn 2 (pipeline)
5. **Demo & Kết luận** - Demo live, đóng góp, hạn chế

## Tùy chỉnh

### Thay đổi theme
```latex
\usetheme{Madrid}      % Có thể đổi thành: Berlin, Warsaw, Singapore, etc.
\usecolortheme{whale}  % Có thể đổi thành: beaver, crane, dolphin, etc.
```

### Thêm logo
Uncomment dòng sau trong `main.tex`:
```latex
\logo{\includegraphics[height=0.8cm]{images/logo.png}}
```

### Thay đổi thông tin
Chỉnh sửa phần:
```latex
\author{Tên sinh viên}
\institute{Tên trường}
\date{Tháng/Năm}
```

## Ghi chú
- Slide được thiết kế với tỉ lệ 16:9 (`aspectratio=169`)
- Font size mặc định: 10pt
- Có 2 backup slides ở cuối cho phần Q&A
