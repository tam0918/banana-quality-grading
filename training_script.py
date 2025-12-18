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

import argparse
import random
from pathlib import Path
from typing import List

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


def _write_yolo_label(path: Path, cls_id: int, xc: float, yc: float, w: float, h: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def create_smoke_dataset(root: Path) -> Path:
    """Create a tiny YOLOv8 dataset for quick pipeline validation.

    This is NOT for accuracy; it only proves training runs.
    """
    import numpy as np
    import cv2

    (root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (root / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    names: List[str] = ["fresh", "ripe", "overripe", "rotten"]
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "names:",
                *[f"  {i}: {n}" for i, n in enumerate(names)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    def gen_one(split: str, idx: int) -> None:
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (20, 20, 20)

        # Draw a fake "banana" ellipse
        cls_id = random.randint(0, 3)
        center = (random.randint(180, 460), random.randint(160, 320))
        axes = (random.randint(120, 170), random.randint(40, 70))
        angle = random.randint(-35, 35)
        if cls_id == 0:
            color = (0, 180, 0)  # green-ish
        elif cls_id == 1:
            color = (0, 220, 220)  # yellow-ish
        elif cls_id == 2:
            color = (0, 190, 220)  # darker yellow
        else:
            color = (20, 20, 20)  # rotten: blend to dark
        cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)

        # Add some random spots for overripe/rotten
        if cls_id in (2, 3):
            for _ in range(35 if cls_id == 2 else 60):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                r = random.randint(2, 8)
                cv2.circle(img, (x, y), r, (0, 0, 0), -1)

        img_path = root / split / "images" / f"smoke_{idx}.jpg"
        cv2.imwrite(str(img_path), img)

        # Approx bbox for the ellipse (rough)
        x1 = max(0, center[0] - axes[0])
        y1 = max(0, center[1] - axes[1])
        x2 = min(w - 1, center[0] + axes[0])
        y2 = min(h - 1, center[1] + axes[1])
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        label_path = root / split / "labels" / f"smoke_{idx}.txt"
        _write_yolo_label(label_path, cls_id, xc, yc, bw, bh)

    for i in range(6):
        gen_one("train", i)
    for i in range(2):
        gen_one("valid", i)

    return data_yaml


def _pick_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 banana quality detector")
    parser.add_argument(
        "--data",
        default="",
        help="Path to data.yaml (if empty: Roboflow env -> datasets/data.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1 ...")
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. Use -1 for Ultralytics AutoBatch (recommended on GPU).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers. Reduce if you hit dataloader issues.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early-stopping patience (epochs).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Create a tiny synthetic dataset and train 1 epoch (pipeline test)",
    )
    args = parser.parse_args()

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

    if args.smoke:
        print("[INFO] Running SMOKE training (synthetic dataset, very small).")
        smoke_root = Path("datasets_smoke")
        data_yaml = Path(create_smoke_dataset(smoke_root))
        args.epochs = 1
        args.imgsz = min(args.imgsz, 320)
    else:
        data_yaml = Path(args.data) if args.data else Path(resolve_data_yaml())

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {data_yaml}.\n"
            "Cách 1: Giải nén dataset YOLOv8 vào thư mục datasets/ để có datasets/data.yaml\n"
            "Cách 2: Set ROBOFLOW_* env vars để tự tải dataset từ Roboflow."
        )

    # Base model nhỏ nhất để train nhanh cho demo (Nano)
    model = YOLO("yolov8n.pt")

    # Train
    device = _pick_device(str(args.device))
    print(f"[INFO] Training with device={device}, epochs={args.epochs}, imgsz={args.imgsz}")

    model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        device=device,
        batch=int(args.batch),
        workers=int(args.workers),
        patience=int(args.patience),
        project="runs_banana",
        name="yolov8n_banana",
    )

    # Sau khi train xong, weights thường nằm ở:
    # runs_banana/yolov8n_banana/weights/best.pt
    print("\nOK. Tìm best.pt tại: runs_banana/yolov8n_banana/weights/best.pt")


if __name__ == "__main__":
    main()
