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
import shutil

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

    # Local fallback for REAL detector dataset (YOLOv8 detection format)
    return str(Path("datasets") / "detector" / "data.yaml")


def _write_yolo_label(path: Path, cls_id: int, xc: float, yc: float, w: float, h: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


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
        "--copy-to-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy trained best.pt to weights/detector.pt for the app (default: true).",
    )
    parser.add_argument(
        "--weights-out",
        default=str(Path("weights") / "detector.pt"),
        help="Destination path when --copy-to-weights is enabled. Default: weights/detector.pt",
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

    data_yaml = Path(args.data) if args.data else Path(resolve_data_yaml())

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {data_yaml}.\n"
            "Cách 1: Giải nén dataset YOLOv8 vào thư mục datasets/detector/ để có datasets/detector/data.yaml\n"
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

    best = Path("runs_banana") / "yolov8n_banana" / "weights" / "best.pt"
    print(f"\n[OK] Best weights: {best}")

    if args.copy_to_weights:
        dst = Path(args.weights_out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(best, dst)
            print(f"[OK] Copied -> {dst}")
        except Exception as e:
            print(f"[WARN] Không copy được detector best.pt sang {dst}: {type(e).__name__}: {e}")
    else:
        print("[INFO] --no-copy-to-weights: giữ weights trong runs_banana/.")


if __name__ == "__main__":
    main()
