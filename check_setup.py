from __future__ import annotations

import os
from pathlib import Path


def _exists(p: str) -> bool:
    try:
        return Path(p).exists() and Path(p).stat().st_size > 0
    except Exception:
        return False


def main() -> int:
    print("Banana Quality Grading — Setup Check\n")

    classifier = str(Path("weights") / "best.pt")
    detector_custom = str(Path("weights") / "detector.pt")
    detector_coco = "yolov8n.pt"
    font = str(Path("assets") / "fonts" / "Roboto-Regular.ttf")
    data_yaml = str(Path("datasets") / "data.yaml")

    print("[Models]")
    print(f"- Classifier (required): {classifier} -> {'OK' if _exists(classifier) else 'MISSING'}")
    print(
        f"- Detector (optional custom): {detector_custom} -> {'OK' if _exists(detector_custom) else 'MISSING'}"
    )
    print(f"- Detector fallback (COCO): {detector_coco} -> {'OK' if _exists(detector_coco) else 'MISSING'}")

    print("\n[Assets]")
    print(f"- Vietnamese font (optional): {font} -> {'OK' if _exists(font) else 'MISSING'}")
    print(f"- datasets/data.yaml (optional mapping): {data_yaml} -> {'OK' if _exists(data_yaml) else 'MISSING'}")

    print("\n[What this means]")
    if _exists(classifier):
        print("- App có thể chạy classifier.")
    else:
        print("- App chưa chạy được: thiếu classifier weights.")

    if _exists(detector_custom):
        print("- App sẽ ưu tiên dùng detector custom (weights/detector.pt).")
    elif _exists(detector_coco):
        print("- App sẽ dùng detector COCO yolov8n.pt (không cần train detector).")
    else:
        print("- App không có detector để tìm bbox chuối.")

    print("\n[Suggested next steps]")
    py = os.environ.get("PY", "")
    if not py:
        # helpful hint for this repo's venv
        py = str(Path(".venv") / "Scripts" / "python.exe")

    if not _exists(classifier):
        print(f"- Train classifier Kaggle: {py} training_kaggle_classification.py --device auto")

    if not _exists(detector_custom):
        print("- Nếu bạn CHƯA có dataset bbox: cứ dùng yolov8n.pt là đủ demo.")
        print(f"- Nếu bạn CÓ dataset bbox YOLOv8: {py} training_script.py --device auto --epochs 50 --imgsz 640")

    if not _exists(font):
        print("- (Tuỳ chọn) Copy font .ttf vào assets/fonts/Roboto-Regular.ttf để hiện tiếng Việt đẹp khi build exe.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
