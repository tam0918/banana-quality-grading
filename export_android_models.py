"""Export YOLO models to Android-friendly formats.

This project uses Ultralytics YOLO (typically backed by PyTorch) which is not practical
to ship inside an Android APK as-is.

The typical mobile path is:
- Export detector + classifier to TensorFlow Lite (TFLite)
- Build a native Android app (Kotlin/Java) that runs TFLite inference (CameraX + overlay)

Usage:
  python export_android_models.py --classifier weights/best.pt --detector yolov8n.pt --imgsz 416

Outputs land under ./exports_android by default.

Notes:
- TFLite export may require extra packages (tensorflow, onnx, onnxsim, etc.) depending
  on your Ultralytics version and export route.
- INT8 quantization generally needs a calibration dataset; this script defaults to float.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is required for export. Install with: pip install ultralytics\n"
            f"Import error: {type(e).__name__}: {e}"
        )


def export_one(
    yolo_cls,
    model_path: str,
    out_dir: Path,
    imgsz: int,
    half: bool,
    int8: bool,
) -> Path:
    model_path = str(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = yolo_cls(model_path)

    # Ultralytics export API varies a bit across versions, but these kwargs are commonly supported.
    # We keep args conservative; if your version errors, remove unsupported kwargs.
    export_kwargs = {
        "format": "tflite",
        "imgsz": imgsz,
        "half": half,
        "int8": int8,
    }

    # Some versions don't accept 'half' or 'int8'. Try progressively.
    last_err: Exception | None = None
    for keys_to_try in [
        ("format", "imgsz", "half", "int8"),
        ("format", "imgsz", "half"),
        ("format", "imgsz"),
        ("format",),
    ]:
        try:
            kwargs = {k: export_kwargs[k] for k in keys_to_try}
            result = model.export(**kwargs)

            # Ultralytics returns different result objects/strings across versions.
            if isinstance(result, str):
                return Path(result)

            # common pattern: dict with 'file'
            if isinstance(result, dict) and "file" in result:
                return Path(str(result["file"]))

            # fallback: search in model's export directory
            # (Ultralytics usually writes alongside the weights or under runs/)
            # We'll just look in the output folder if possible.
            candidates = list(out_dir.glob("**/*.tflite"))
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return candidates[0]

            # last resort: no path detected
            raise RuntimeError("Export succeeded but output file not found")
        except Exception as e:
            last_err = e

    raise SystemExit(f"Export failed for {model_path}: {type(last_err).__name__}: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export detector/classifier to TFLite for Android")
    parser.add_argument(
        "--classifier",
        default=os.path.join("weights", "best.pt"),
        help="Path to classifier weights (.pt). Default: weights/best.pt",
    )
    parser.add_argument(
        "--detector",
        default="yolov8n.pt",
        help="Path to detector weights (.pt). Default: yolov8n.pt",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Export image size (square). Common: 320/416/640",
    )
    parser.add_argument(
        "--outdir",
        default="exports_android",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Try FP16 where supported (may reduce size)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Try INT8 quantization (usually needs calibration; may fail without it)",
    )

    args = parser.parse_args()

    YOLO = _require_ultralytics()
    out_dir = Path(args.outdir)

    print("Exporting detector to TFLite...")
    det_tflite = export_one(YOLO, args.detector, out_dir / "detector", args.imgsz, args.half, args.int8)
    print(f"Detector exported: {det_tflite}")

    print("Exporting classifier to TFLite...")
    cls_tflite = export_one(YOLO, args.classifier, out_dir / "classifier", args.imgsz, args.half, args.int8)
    print(f"Classifier exported: {cls_tflite}")

    print("\nNext step: build Android app (Kotlin) and run these TFLite models via CameraX.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
