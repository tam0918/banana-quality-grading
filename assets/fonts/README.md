# Fonts (Vietnamese)

App vẽ chữ tiếng Việt bằng Pillow + font `.ttf`.

## Cách dùng nhanh (Windows)
- Không cần copy font vào đây nếu máy bạn có sẵn font hệ thống.
- Mặc định app sẽ thử các font:
  - `assets/fonts/Roboto-Regular.ttf`
  - `C:/Windows/Fonts/segoeui.ttf`
  - `C:/Windows/Fonts/arial.ttf`

## Khuyến nghị khi build .exe để chạy trên máy khác
- Đặt 1 file font `.ttf` hỗ trợ tiếng Việt vào thư mục này và đặt tên `Roboto-Regular.ttf`.
- Hoặc set env `BANANA_FONT_URL` để app tự tải font về `assets/fonts/Roboto-Regular.ttf`.

Lưu ý: repo không kèm sẵn font để tránh vấn đề bản quyền/phân phối.
