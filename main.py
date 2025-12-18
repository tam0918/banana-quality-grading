from __future__ import annotations

import os

from app.ui_manager import UIConfig, UI_Manager
from utils.resource_manager import ResourceManager

# ================================
# Font tiếng Việt (TTF)
# ================================
# OpenCV cv2.putText không hiển thị Unicode tiếng Việt ổn định.
# App này dùng Pillow + ImageDraw để vẽ chữ tiếng Việt lên video frame.
#
# CÁCH THIẾT LẬP:
# 1) (Khuyến nghị) Copy font hỗ trợ tiếng Việt vào: assets/fonts/Roboto-Regular.ttf
#    Ví dụ: Roboto, NotoSans, Segoe UI.
# 2) Hoặc trỏ tới font hệ thống Windows:
#    - C:/Windows/Fonts/arial.ttf
#    - C:/Windows/Fonts/segoeui.ttf
#
# Nếu không tìm thấy font, app sẽ fallback sang English và in cảnh báo.

DEFAULT_FONT_CANDIDATES = [
    os.path.join("assets", "fonts", "Roboto-Regular.ttf"),
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


# ================================
# YOLOv8 weights (best.pt)
# ================================
# Sau khi train bằng training_script.py, bạn sẽ có file:
#   runs_banana/yolov8n_banana/weights/best.pt
#
# Bạn có thể:
# - Copy best.pt về thư mục `weights/best.pt` (khuyến nghị)
# - Hoặc chỉnh danh sách DEFAULT_MODEL_CANDIDATES bên dưới

DEFAULT_MODEL_CANDIDATES = [
    os.path.join("weights", "best.pt"),
    os.path.join("runs_banana", "yolov8n_banana", "weights", "best.pt"),
]


# ================================
# YOLOv8 detector weights
# ================================
# App uses a 2-stage pipeline:
# - Detector finds banana bbox
# - Classifier predicts ripeness on the crop
#
# Default detector is COCO pretrained (yolov8n.pt).
# If you train your own banana detector, put it at:
#   weights/detector.pt
# or set env BANANA_DETECTOR_PATH.

DEFAULT_DETECTOR_CANDIDATES = [
    os.path.join("weights", "detector.pt"),
    os.path.join("runs_banana", "yolov8n_banana", "weights", "best.pt"),
    "yolov8n.pt",
]


# ================================
# Auto-provisioning (tải tài nguyên tự động)
# ================================
# Bạn có thể upload best.pt + font lên GitHub Releases rồi dán link tải trực tiếp.
#
# Cách dùng nhanh (khuyến nghị): set ENV thay vì sửa code:
#   setx BANANA_MODEL_URL "https://.../best.pt"
#   setx BANANA_FONT_URL  "https://.../Roboto-Regular.ttf"
#
# Hoặc sửa 2 biến dưới đây.

MODEL_URL = os.environ.get("BANANA_MODEL_URL", "")  # TODO: dán link trực tiếp best.pt
FONT_URL = os.environ.get("BANANA_FONT_URL", "")    # TODO: dán link trực tiếp font .ttf
DETECTOR_URL = os.environ.get("BANANA_DETECTOR_URL", "")


def pick_font_path() -> str:
    for p in DEFAULT_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    # Return first candidate; renderer will warn + fallback.
    return DEFAULT_FONT_CANDIDATES[0]


def pick_model_path() -> str:
    for p in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    # Return first candidate; grader will show message if missing.
    return DEFAULT_MODEL_CANDIDATES[0]


def pick_detector_path() -> str:
    env_path = os.environ.get("BANANA_DETECTOR_PATH", "").strip()
    if env_path:
        return env_path
    for p in DEFAULT_DETECTOR_CANDIDATES:
        if os.path.exists(p):
            return p
    return DEFAULT_DETECTOR_CANDIDATES[-1]


def main() -> None:
    # Download missing resources if URLs provided.
    rm = ResourceManager()
    rm.check_and_download(os.path.join("weights", "best.pt"), MODEL_URL, "YOLO weights (best.pt)")
    rm.check_and_download(os.path.join("assets", "fonts", "Roboto-Regular.ttf"), FONT_URL, "Vietnamese font (TTF)")
    rm.check_and_download(os.path.join("weights", "detector.pt"), DETECTOR_URL, "YOLO detector weights (detector.pt)")

    ui = UI_Manager(
        UIConfig(
            font_path=pick_font_path(),
            model_path=pick_model_path(),
            detector_model_path=pick_detector_path(),
            camera_index=0,
        )
    )
    ui.run()


if __name__ == "__main__":
    main()
