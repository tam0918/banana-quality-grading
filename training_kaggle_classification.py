from __future__ import annotations

"""Train YOLOv8 Classification model using Kaggle dataset.

Dataset target (user request):
  https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset

This Kaggle dataset is IMAGE CLASSIFICATION (folders per class), not detection.
So we train a YOLOv8 *classifier* (yolov8n-cls.pt) and later combine it with a
YOLOv8 *detector* (COCO banana) at runtime to draw a bounding box.

Auto-download options:
1) Kaggle API (recommended)
    - Put kaggle.json at: %USERPROFILE%/.kaggle/kaggle.json
     OR set env: KAGGLE_USERNAME, KAGGLE_KEY
   - Then run this script; it will download/unzip if missing.

2) Manual download
   - Download from Kaggle and unzip into datasets/kaggle_banana_ripeness/

Run:
  python training_kaggle_classification.py

Output weights:
  runs_banana/yolov8n_banana_cls/weights/best.pt
Copy to:
  weights/best.pt   (app default)
"""

import argparse
import os
import subprocess
import shutil
import zipfile
from pathlib import Path
import sys

from ultralytics import YOLO


KAGGLE_DATASET = "shahriar26s/banana-ripeness-classification-dataset"
TARGET_DIR = Path("datasets") / "kaggle_banana_ripeness"
ZIP_PATH = Path("datasets") / "kaggle_banana_ripeness.zip"


def _maybe_kaggle_download() -> None:
    """Download dataset zip via Kaggle API if not already present.

    Kaggle does not allow anonymous downloads; credentials required.
    """
    if TARGET_DIR.exists() and any(TARGET_DIR.rglob("*.jpg")):
        print(f"[OK] Found dataset folder: {TARGET_DIR}")
        return

    # Reuse any existing zip in datasets/ (common when user downloaded manually)
    if not ZIP_PATH.exists():
        zips = sorted(ZIP_PATH.parent.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        for z in zips:
            # Prefer the expected dataset zip name if present
            if "banana-ripeness-classification-dataset" in z.name:
                shutil.move(str(z), str(ZIP_PATH))
                break
        if not ZIP_PATH.exists() and zips:
            # Fallback: take newest zip
            shutil.move(str(zips[0]), str(ZIP_PATH))

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0:
        print(f"[OK] Found dataset zip: {ZIP_PATH}")
        need_download = False
    else:
        need_download = True

    if need_download:
        print("[INFO] Dataset not found locally. Trying Kaggle download...")
        try:
            ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Kaggle requires credentials. Setup options:
            # - %USERPROFILE%\.kaggle\kaggle.json
            # - or env vars: KAGGLE_USERNAME, KAGGLE_KEY
            #
            # Use CLI to download (stable across kaggle versions).
            kaggle_exe = str(Path(sys.executable).with_name("kaggle.exe"))
            cmd = [
                kaggle_exe,
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                str(ZIP_PATH.parent),
                "--force",
            ]
            print(f"[INFO] Running: {' '.join(cmd)}")
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout).strip())

            # Kaggle creates a zip named after the dataset slug.
            zips = sorted(ZIP_PATH.parent.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not zips:
                raise FileNotFoundError("Kaggle download did not produce a zip file.")

            newest = zips[0]
            if newest != ZIP_PATH:
                shutil.move(str(newest), str(ZIP_PATH))

            print(f"[OK] Downloaded zip: {ZIP_PATH}")
        except Exception as e:
            raise RuntimeError(
                "Không thể tải dataset từ Kaggle (cần credentials).\n"
                "Cách fix: đặt kaggle.json ở %USERPROFILE%/.kaggle/kaggle.json\n"
                "hoặc set KAGGLE_USERNAME + KAGGLE_KEY.\n"
                f"Chi tiết lỗi: {type(e).__name__}: {e}"
            )

    # Unzip
    if TARGET_DIR.exists():
        # If folder exists but empty/broken, remove
        if not any(TARGET_DIR.rglob("*")):
            shutil.rmtree(TARGET_DIR, ignore_errors=True)

    if not TARGET_DIR.exists():
        print(f"[INFO] Unzipping to {TARGET_DIR} ...")
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(TARGET_DIR)

    # Some Kaggle zips may contain a nested folder; try to locate train/valid/test
    if not (TARGET_DIR / "train").exists():
        candidates = [p for p in TARGET_DIR.glob("**/train") if p.is_dir()]
        if candidates:
            root = candidates[0].parent
            print(f"[INFO] Detected nested dataset root: {root}")
            # Move contents up to TARGET_DIR
            for item in root.iterdir():
                dest = TARGET_DIR / item.name
                if dest.exists():
                    continue
                shutil.move(str(item), str(dest))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 classifier from Kaggle banana ripeness dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
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
        default=None,
        help=(
            "Dataloader workers. On Windows, use 0-2 if you hit RAM/DataLoader issues. "
            "If omitted, the script chooses a safe default for your OS."
        ),
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
        help="Copy trained best.pt to weights/best.pt for the app (default: true).",
    )
    parser.add_argument(
        "--weights-out",
        default=os.path.join("weights", "best.pt"),
        help="Destination path when --copy-to-weights is enabled. Default: weights/best.pt",
    )
    parser.add_argument("--download-only", action="store_true", help="Only download+unzip dataset, skip training")
    args = parser.parse_args()

    # Windows DataLoader uses multiprocessing with high per-worker overhead.
    # Defaulting to many workers can easily exhaust system RAM and crash with
    # `DefaultCPUAllocator: not enough memory`, even for small allocations.
    if args.workers is None:
        args.workers = 0 if os.name == "nt" else 8

    # Ensure dataset exists
    _maybe_kaggle_download()

    # Ultralytics classification expects:
    #   dataset_root/train/<class>/*.jpg
    #   dataset_root/val/<class>/*.jpg
    # Kaggle dataset uses "valid" not "val"; we create a "val" alias.
    if (TARGET_DIR / "valid").exists() and not (TARGET_DIR / "val").exists():
        try:
            os.symlink(str((TARGET_DIR / "valid").resolve()), str((TARGET_DIR / "val").resolve()))
        except Exception:
            # Windows without admin may fail symlink; just copy folder name mapping
            (TARGET_DIR / "val").mkdir(parents=True, exist_ok=True)
            # Shallow copy by creating junction-like structure is complex; do a simple copy on first run.
            # (Dataset is not too large ~230MB, acceptable.)
            if not any((TARGET_DIR / "val").rglob("*.jpg")):
                print("[INFO] Copying valid/ -> val/ (one-time) ...")
                shutil.copytree(TARGET_DIR / "valid", TARGET_DIR / "val", dirs_exist_ok=True)

    if not (TARGET_DIR / "train").exists() or not (TARGET_DIR / "val").exists():
        raise FileNotFoundError(
            f"Dataset structure invalid under {TARGET_DIR}. Expected train/ and val/ folders."
        )

    if args.download_only:
        print("[OK] Download-only mode. Dataset is ready.")
        return

    model = YOLO("yolov8n-cls.pt")

    device = str(args.device)
    if device == "auto":
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"[INFO] Training classifier on {TARGET_DIR} with device={device}")

    try:
        model.train(
            data=str(TARGET_DIR),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            device=device,
            batch=int(args.batch),
            workers=int(args.workers),
            patience=int(args.patience),
            project="runs_banana",
            name="yolov8n_banana_cls",
        )
    except RuntimeError as e:
        msg = str(e)
        if "DefaultCPUAllocator" in msg and "not enough memory" in msg:
            raise RuntimeError(
                "Training failed due to insufficient system RAM during DataLoader preprocessing.\n"
                "Fix options (pick one):\n"
                "  1) Reduce dataloader workers: --workers 0 (Windows-safe)\n"
                "  2) Reduce image size: --imgsz 224 (typical for classification)\n"
                "  3) Reduce batch size (if not using AutoBatch): --batch 16\n"
                "Also check Windows Virtual Memory (pagefile) is enabled.\n"
                f"Original error: {e}"
            )
        raise

    best = Path("runs_banana") / "yolov8n_banana_cls" / "weights" / "best.pt"
    print(f"\n[OK] Best weights: {best}")

    if args.copy_to_weights:
        dst = Path(args.weights_out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(best, dst)
            print(f"[OK] Copied -> {dst}")
        except Exception as e:
            print(f"[WARN] Không copy được best.pt sang {dst}: {type(e).__name__}: {e}")
    else:
        print("[INFO] --no-copy-to-weights: giữ weights trong runs_banana/.")


if __name__ == "__main__":
    main()
