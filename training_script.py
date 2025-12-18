from __future__ import annotations

import os

# ================================
# YOLOv8 Training Script (Transfer Learning)
# ================================
# Yêu cầu dataset theo chuẩn YOLO:
# - Có file data.yaml
# - Có thư mục images/ và labels/ cho train/val (và test nếu có)
#
# Ví dụ cấu trúc phổ biến:
# dataset/
#   data.yaml
#   train/images/*.jpg
#   train/labels/*.txt
#   val/images/*.jpg
#   val/labels/*.txt
#
# Gợi ý nguồn dataset:
# - Roboflow Universe (đã có bbox, xuất YOLOv8 format)
#
# Chạy:
#   python training_script.py

from pathlib import Path

from ultralytics import YOLO


def resolve_data_yaml() -> str:
    """Resolve dataset data.yaml path.

    Priority:
    1) If Roboflow env vars are provided, auto-download dataset (YOLOv8 format)
    2) Else, use local: datasets/data.yaml

    Roboflow env vars (recommended; avoids hardcoding secrets in code):
    - ROBOFLOW_API_KEY
    - ROBOFLOW_WORKSPACE (workspace slug)
    - ROBOFLOW_PROJECT (project slug)
    - ROBOFLOW_VERSION (int, default 1)
    """

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    ws = os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
    proj = os.environ.get("ROBOFLOW_PROJECT", "").strip()
    ver = os.environ.get("ROBOFLOW_VERSION", "1").strip() or "1"

    if api_key and ws and proj:
        try:
            from roboflow import Roboflow

            print("[INFO] Roboflow auto-download enabled.")
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(ws).project(proj)
            dataset = project.version(int(ver)).download("yolov8")

            data_yaml = Path(dataset.location) / "data.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(f"Roboflow dataset downloaded but missing data.yaml at {data_yaml}")

            print(f"[OK] Dataset ready at: {dataset.location}")
            return str(data_yaml)
        except Exception as e:
            print(f"[WARN] Roboflow download failed, fallback to local datasets/: {type(e).__name__}: {e}")

    # Local fallback
    return str(Path("datasets") / "data.yaml")


def main() -> None:
    # ================================
    # Auto-download dataset (optional)
    # ================================
    # Nếu bạn muốn code tự tải dataset từ Roboflow (không tải zip thủ công),
    # hãy set ENV trước khi chạy:
    #   setx ROBOFLOW_API_KEY "<YOUR_API_KEY>"
    #   setx ROBOFLOW_WORKSPACE "<workspace-slug>"
    #   setx ROBOFLOW_PROJECT "<project-slug>"
    #   setx ROBOFLOW_VERSION "1"
    #
    # Nếu không set env, script sẽ dùng đường dẫn local: datasets/data.yaml

    data_yaml = Path(resolve_data_yaml())

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {data_yaml}.\n"
            "Cách 1: Giải nén dataset YOLOv8 vào thư mục datasets/ để có datasets/data.yaml\n"
            "Cách 2: Set ROBOFLOW_* env vars để tự tải dataset từ Roboflow."
        )

    # Base model nhỏ nhất để train nhanh cho demo (Nano)
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        device=0,  # đổi thành 'cpu' nếu không có GPU
        project="runs_banana",
        name="yolov8n_banana",
    )

    # Sau khi train xong, weights thường nằm ở:
    # runs_banana/yolov8n_banana/weights/best.pt
    print("\nOK. Tìm best.pt tại: runs_banana/yolov8n_banana/weights/best.pt")


if __name__ == "__main__":
    main()
