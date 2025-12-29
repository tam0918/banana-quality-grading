from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import json

import numpy as np
import pytest

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:  # pragma: no cover
    HAS_CV2 = False

from app.grader import BananaGrader


ARTIFACT_DIR = Path("tests/artifacts")


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _write_artifacts(prefix: str, frame_bgr: np.ndarray, result, sample_img: Path, cls_w: str, det_w: str) -> None:
    """Write artifacts to help debug what the real models produced."""
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        payload = {
            "sample_image": str(sample_img),
            "classifier_weights": cls_w,
            "detector_weights": det_w,
            "category_key": getattr(result, "category_key", None),
            "label_vi": getattr(result, "label_vi", None),
            "label_en": getattr(result, "label_en", None),
            "confidence": float(getattr(result, "confidence", 0.0)),
            "bbox_xyxy": getattr(result, "bbox_xyxy", None),
            "quality_score": float(getattr(result, "quality_score", 0.0)),
            "spot_count": int(getattr(result, "spot_count", 0)),
            "refined": bool(getattr(result, "refined", False)),
            "debug": getattr(result, "debug", {}),
            "color_features": getattr(result, "color_features", None),
        }

        json_path = ARTIFACT_DIR / f"{prefix}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if HAS_CV2:
            out = frame_bgr.copy()
            h, w = out.shape[:2]
            bbox = getattr(result, "bbox_xyxy", None)
            if bbox is not None:
                x1, y1, x2, y2 = _clamp_bbox(*bbox, w=w, h=h)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    out,
                    f"{payload['category_key']} {payload['confidence']*100:.0f}% Q={payload['quality_score']*100:.0f}%",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            img_path = ARTIFACT_DIR / f"{prefix}.jpg"
            cv2.imwrite(str(img_path), out)
    except Exception:
        # Artifacts are best-effort; tests should still report the real failure.
        return


def _pick_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _load_bgr(path: Path) -> np.ndarray:
    if not HAS_CV2:
        pytest.fail("OpenCV not installed (cv2). Install opencv-python to run real-model integration tests.")
    img = cv2.imread(str(path))
    if img is None:
        pytest.fail(f"Could not read image: {path}")
    return img


def _candidate_sample_images() -> list[Path]:
    # These images are produced by Ultralytics training runs (mosaics/val previews)
    # and should be safe to use as deterministic inputs for integration tests.
    return [
        Path("tests/assets/banana.jpg"),
        Path("runs_banana/yolov8n_banana_cls3/val_batch0_labels.jpg"),
        Path("runs_banana/yolov8n_banana_cls3/val_batch0_pred.jpg"),
        Path("runs_banana/yolov8n_banana_cls3/train_batch0.jpg"),
        Path("runs_banana/yolov8n_banana_cls/train_batch0.jpg"),
        Path("runs_banana/yolov8n_banana_cls2/train_batch0.jpg"),
    ]


def _pick_real_weights() -> tuple[str, str]:
    # classifier candidates
    cls_candidates = [
        Path("yolov8n_banana_cls3_best.pt"),
        Path("weights/best.pt"),
        Path("runs_banana/yolov8n_banana_cls3/weights/best.pt"),
        Path("runs_banana/yolov8n_banana_cls/weights/best.pt"),
    ]
    detector_candidates = [
        Path("weights/detector.pt"),
        Path("yolov8n.pt"),
    ]

    cls = _pick_first_existing(cls_candidates)
    det = _pick_first_existing(detector_candidates)

    if cls is None:
        pytest.fail(
            "Classifier weights not found. Expected one of: "
            + ", ".join(str(p) for p in cls_candidates)
        )
    if det is None:
        pytest.fail(
            "Detector weights not found. Expected one of: "
            + ", ".join(str(p) for p in detector_candidates)
        )

    return str(cls), str(det)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
def test_grader_runs_end_to_end_with_real_models_cpu():
    sample_img = Path("tests/assets/banana.jpg")
    if not sample_img.exists():
        pytest.fail("Missing required test image: tests/assets/banana.jpg")

    cls_w, det_w = _pick_real_weights()

    frame = _load_bgr(sample_img)

    grader = BananaGrader(
        model_path=cls_w,
        detector_model_path=det_w,
        device="cpu",
        conf=0.25,
        iou=0.45,
        enable_enhanced_analysis=True,
        enable_preprocessing=True,
        enable_feature_refinement=True,
    )

    result = grader.grade(frame)
    _write_artifacts("real_e2e", frame, result, sample_img, cls_w, det_w)

    # With a dedicated banana image (tests/assets/banana.jpg), we expect detection to succeed.
    assert result.bbox_xyxy is not None, (
        f"Detector failed to find banana in {sample_img}. "
        "See tests/artifacts/real_e2e.json and tests/artifacts/real_e2e.jpg"
    )
    assert result.category_key != "none", (
        f"Detector/classifier returned none for {sample_img}. "
        "See tests/artifacts/real_e2e.json and tests/artifacts/real_e2e.jpg"
    )

    # Additional quality checks to point out what is 'unfinished' when it fails.
    x1, y1, x2, y2 = result.bbox_xyxy
    ih, iw = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, w=iw, h=ih)
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    img_area = iw * ih
    assert box_area >= img_area * 0.03, (
        f"Detected bbox too small ({box_area/img_area:.2%} of image). "
        "Likely wrong detection. See tests/artifacts/real_e2e.jpg"
    )

    assert result.category_key in {"unripe", "export", "overripe", "defective"}
    assert 0.0 <= result.confidence <= 1.0

    # Enhanced analysis fields should be populated when enabled
    assert 0.0 <= result.quality_score <= 1.0
    assert isinstance(result.spot_count, int)
    assert result.quality_score > 0.05, (
        "Enhanced analysis produced near-zero quality score. "
        "Check lighting/background or HSV thresholds. See tests/artifacts/real_e2e.json"
    )


@pytest.mark.integration
@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
def test_grader_classifier_stage_runs_on_crop_even_if_forced_bbox():
    """Classifier + analyzer integration test that does not depend on the detector.

    This test ensures we still validate the *real classifier weights* even if detection
    is unstable on the chosen sample image.
    """

    sample_img = Path("tests/assets/banana.jpg")
    if not sample_img.exists():
        pytest.fail("Missing required test image: tests/assets/banana.jpg")

    cls_w, det_w = _pick_real_weights()

    frame = _load_bgr(sample_img)

    grader = BananaGrader(
        model_path=cls_w,
        detector_model_path=det_w,
        device="cpu",
        enable_enhanced_analysis=True,
        enable_preprocessing=True,
        enable_feature_refinement=True,
    )

    # Force a bbox = whole image by seeding the bbox-hold cache.
    h, w = frame.shape[:2]
    grader._update_last_bbox((0, 0, w - 1, h - 1))
    grader._bbox_hold_remaining = max(1, grader._bbox_hold_remaining)

    # Make the detector intentionally miss once by setting hold-only mode.
    # We can do that by using a too-high conf temporarily.
    grader.conf = 0.99

    result = grader.grade(frame)
    _write_artifacts("real_cls_holdbbox", frame, result, sample_img, cls_w, det_w)

    # Because bbox-hold is enabled by default, we should still get a classification.
    assert result.bbox_xyxy is not None, "Expected bbox via hold-bbox fallback. See tests/artifacts/real_cls_holdbbox.json"
    assert result.category_key in {"unripe", "export", "overripe", "defective", "none"}

    # If classifier fails to run, category_key would be none with error.
    assert "error" not in result.debug, (
        "Classifier/analyzer failed (error in debug). "
        "See tests/artifacts/real_cls_holdbbox.json"
    )

    # Enhanced analysis should run on the crop
    assert 0.0 <= result.quality_score <= 1.0
    assert result.quality_score > 0.0, "Expected analyzer to produce a non-zero score. See tests/artifacts/real_cls_holdbbox.json"
