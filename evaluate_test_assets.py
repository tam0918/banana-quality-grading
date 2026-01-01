from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from app.grader import BananaGrader

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


def _iter_images(images_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _load_bgr(path: Path) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image via OpenCV: {path}")
        return img

    # Fallback: PIL
    from PIL import Image  # type: ignore

    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    # RGB -> BGR
    return arr[:, :, ::-1].copy()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _draw_annotation(
    frame_bgr: np.ndarray,
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
    text: str,
) -> np.ndarray:
    out = frame_bgr.copy()
    if cv2 is None:
        return out

    h, w = out.shape[:2]
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put label at top-left (or inside bbox if present)
    org_x, org_y = 10, 30
    if bbox_xyxy is not None:
        org_x, org_y = max(10, bbox_xyxy[0] + 5), max(30, bbox_xyxy[1] + 25)

    cv2.putText(out, text, (org_x, org_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


def _grade_color_bgr(key: str) -> Tuple[int, int, int]:
    # Mirror UI colors: Unripe=Green, Export=Blue, Overripe=Orange, Defective=Red.
    if key == "unripe":
        return (0, 255, 0)
    if key == "export":
        return (255, 0, 0)
    if key == "overripe":
        return (0, 165, 255)
    if key == "defective":
        return (0, 0, 255)
    return (180, 180, 180)


def _draw_multi_annotations(frame_bgr: np.ndarray, items) -> np.ndarray:
    out = frame_bgr.copy()
    if cv2 is None:
        return out

    for it in items or []:
        bbox = getattr(it, "bbox_xyxy", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        color = _grade_color_bgr(getattr(it, "category_key", "none"))
        thickness = 4 if getattr(it, "category_key", "") == "defective" else 3
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        label = f"{getattr(it, 'category_key', 'none')} {float(getattr(it, 'confidence', 0.0)):.2f}"
        ty = max(25, int(y1) - 10)
        cv2.putText(out, label, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return out


def _mapping_from_data_yaml(path: str) -> Optional[dict[int, str]]:
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    names = data.get("names") if isinstance(data, dict) else None
    if not names:
        return None

    if isinstance(names, list):
        ordered = list(names)
    elif isinstance(names, dict):
        ordered = [v for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    else:
        return None

    def normalize(s: str) -> set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in s).strip()
        return set(cleaned.split())

    mapping: dict[int, str] = {}
    for idx, class_name in enumerate(ordered):
        tokens = normalize(str(class_name))

        if tokens.intersection({"rotten", "rot", "bad", "mold", "mould", "disease", "defect", "defective", "spoiled"}):
            mapping[idx] = "defective"
        elif tokens.intersection({"overripe", "over", "brown", "spotted", "spot", "bruise"}):
            mapping[idx] = "overripe"
        elif tokens.intersection({"ripe", "export", "yellow", "just"}):
            mapping[idx] = "export"
        elif tokens.intersection({"unripe", "green", "fresh", "raw"}):
            mapping[idx] = "unripe"

    return mapping if len(mapping) >= 2 else None


def _infer_category_from_class_name(name: str) -> Optional[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(name)).strip()
    tokens = set(cleaned.split())

    if tokens.intersection({"rotten", "rot", "bad", "mold", "mould", "disease", "defect", "defective", "spoiled"}):
        return "defective"
    if tokens.intersection({"overripe", "over", "brown", "spotted", "spot", "bruise"}):
        return "overripe"
    if tokens.intersection({"ripe", "export", "yellow", "just"}):
        return "export"
    if tokens.intersection({"unripe", "green", "fresh", "raw"}):
        return "unripe"

    return None


def _default_classid_to_category() -> dict[int, str]:
    # Match BananaGrader._default_mapping()
    return {0: "unripe", 1: "export", 2: "overripe", 3: "defective"}


def _pick_device(requested: str) -> str:
    req = str(requested or "").strip()
    if not req:
        return "cpu"
    if req.lower() != "auto":
        return req

    try:
        import torch  # type: ignore

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate banana grading model on test assets.")
    parser.add_argument("--images", default="tests/assets", help="Folder containing test images")
    parser.add_argument("--classifier", default="weights/best.pt", help="Classifier weights path")
    parser.add_argument("--detector", default="yolov8n.pt", help="Detector weights path")
    parser.add_argument("--det-conf", type=float, default=0.25, help="Detector confidence threshold (pipeline mode)")
    parser.add_argument("--det-iou", type=float, default=0.45, help="Detector IoU threshold (pipeline mode)")
    parser.add_argument("--data-yaml", default="", help="Optional data.yaml for class-name mapping")
    parser.add_argument("--device", default=os.environ.get("BANANA_DEVICE", "auto"), help="Ultralytics device (cpu, 0, auto)")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "classifier"],
        default="pipeline",
        help="pipeline=detector+bbox+classifier (like app), classifier=run classifier on full image",
    )
    parser.add_argument(
        "--bbox-hold",
        type=int,
        default=0,
        help="Hold last bbox across frames. For batch image eval this should be 0 (default).",
    )
    parser.add_argument("--out", default="tests/artifacts/eval_assets", help="Output directory")
    parser.add_argument("--save-images", action="store_true", help="Save annotated images")
    args = parser.parse_args()

    resolved_device = _pick_device(args.device)

    images_dir = Path(args.images)
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images folder not found: {images_dir}")

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    # Record run metadata
    meta_path = out_dir / "run_meta.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"timestamp={datetime.now().isoformat(timespec='seconds')}",
                f"images={images_dir.resolve()}",
                f"classifier={Path(args.classifier).resolve()}",
                f"detector={Path(args.detector).resolve()}",
                f"data_yaml={args.data_yaml}",
                f"device={args.device}",
                f"device_resolved={resolved_device}",
                f"mode={args.mode}",
                f"bbox_hold={args.bbox_hold}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # The grader is designed for video; disable temporal bbox holding by default so each image is independent.
    os.environ["BANANA_BBOX_HOLD"] = str(int(args.bbox_hold))

    grader: Optional[BananaGrader] = None
    classifier = None
    classid_to_category = _mapping_from_data_yaml(args.data_yaml) if args.data_yaml else None
    if classid_to_category is None:
        classid_to_category = _default_classid_to_category()

    if args.mode == "pipeline":
        grader = BananaGrader(
            model_path=args.classifier,
            detector_model_path=args.detector,
            data_yaml_path=(args.data_yaml or None),
            device=resolved_device,
            conf=float(args.det_conf),
            iou=float(args.det_iou),
        )
    else:
        if YOLO is None:
            raise SystemExit("ultralytics is not installed; cannot run classifier-only mode")
        classifier = YOLO(args.classifier)

    rows = []
    counts: dict[str, int] = {}
    counts_instances: dict[str, int] = {}

    annotated_dir = out_dir / "annotated"
    if args.save_images:
        _ensure_dir(annotated_dir)

    for img_path in _iter_images(images_dir):
        frame_bgr = _load_bgr(img_path)

        if args.mode == "pipeline":
            assert grader is not None
            fg = grader.grade_frame(frame_bgr)
            res = fg.overall

            # Frame-level count (overall decision)
            counts[res.category_key] = counts.get(res.category_key, 0) + 1

            # Per-instance rows
            for idx, it in enumerate(fg.items):
                counts_instances[it.category_key] = counts_instances.get(it.category_key, 0) + 1

                bbox = it.bbox_xyxy
                bbox_str = "" if bbox is None else f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

                row = {
                    "file": img_path.name,
                    "instance": str(idx),
                    "category_key": it.category_key,
                    "label_vi": it.label_vi,
                    "label_en": it.label_en,
                    "status_vi": it.status_vi,
                    "confidence": f"{it.confidence:.4f}",
                    "bbox_xyxy": bbox_str,
                    "quality_score": f"{getattr(it, 'quality_score', 0.0):.4f}",
                    "spot_count": str(getattr(it, 'spot_count', 0)),
                }

                dbg = getattr(it, "debug", {}) or {}
                for k in sorted(dbg.keys()):
                    row[f"debug.{k}"] = f"{float(dbg[k]):.6f}" if isinstance(dbg[k], (float, int)) else str(dbg[k])

                # Enrich with classifier class-name based category inference
                try:
                    cls_id = int(float(dbg.get("cls_id"))) if "cls_id" in dbg else None
                except Exception:
                    cls_id = None
                cls_name = ""
                inferred = ""
                try:
                    clf = getattr(grader, "_classifier", None)
                    names = getattr(clf, "names", {}) if clf is not None else {}
                    if cls_id is not None and isinstance(names, dict):
                        cls_name = str(names.get(cls_id, ""))
                        inferred_cat = _infer_category_from_class_name(cls_name)
                        inferred = inferred_cat or ""
                except Exception:
                    pass
                row["cls_id"] = "" if cls_id is None else str(cls_id)
                row["cls_name"] = cls_name
                row["category_key_by_name"] = inferred

                rows.append(row)

            # If detector found nothing, still emit a row for visibility.
            if not fg.items:
                rows.append(
                    {
                        "file": img_path.name,
                        "instance": "",
                        "category_key": "none",
                        "label_vi": "Không phát hiện chuối",
                        "label_en": "No banana detected",
                        "status_vi": "—",
                        "confidence": f"{0.0:.4f}",
                        "bbox_xyxy": "",
                    }
                )

            if args.save_images and cv2 is not None:
                annotated = _draw_multi_annotations(frame_bgr, fg.items)
                # Also print overall decision at top-left
                overall_label = f"overall={res.category_key} ({res.confidence:.2f})"
                annotated = _draw_annotation(annotated, None, overall_label)
                cv2.imwrite(str(annotated_dir / img_path.name), annotated)

        else:
            assert classifier is not None
            pred = classifier.predict(source=frame_bgr, verbose=False, device=resolved_device)
            if not pred:
                category_key = "none"
                cls_id = ""
                cls_name = ""
                conf = 0.0
            else:
                probs = getattr(pred[0], "probs", None)
                if probs is None:
                    category_key = "none"
                    cls_id = ""
                    cls_name = ""
                    conf = 0.0
                else:
                    top1 = int(getattr(probs, "top1", 0))
                    conf = float(getattr(probs, "top1conf", 0.0))
                    names = getattr(classifier, "names", {}) or {}
                    cls_name = str(names.get(top1, ""))
                    cls_id = str(top1)
                    category_key = _infer_category_from_class_name(cls_name) or classid_to_category.get(top1, "none")

            counts[category_key] = counts.get(category_key, 0) + 1
            row = {
                "file": img_path.name,
                "category_key": category_key,
                "confidence": f"{conf:.4f}",
                "cls_id": cls_id,
                "cls_name": cls_name,
                "bbox_xyxy": "",
                "status_vi": "—",
            }
            rows.append(row)

            if args.save_images and cv2 is not None:
                label = f"{category_key} ({conf:.2f})"
                annotated = _draw_annotation(frame_bgr, None, label)
                cv2.imwrite(str(annotated_dir / img_path.name), annotated)

    # Write CSV
    csv_path = out_dir / "predictions.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write summary
    summary_path = out_dir / "summary.txt"
    lines = ["Counts by category_key (frame-level overall):"]
    for k in sorted(counts.keys()):
        lines.append(f"- {k}: {counts[k]}")
    lines.append("")
    lines.append("Counts by category_key (instance-level):")
    for k in sorted(counts_instances.keys()):
        lines.append(f"- {k}: {counts_instances[k]}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    if args.save_images:
        print(f"Wrote annotated images: {annotated_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
