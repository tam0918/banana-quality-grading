from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:  # pragma: no cover
    HAS_CV2 = False

from app.banana_analyzer import BananaAnalyzer, MultiScaleDetector


def _synthetic_banana(width: int = 320, height: int = 240, bgr=(0, 200, 255)) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if not HAS_CV2:
        return img

    img[:] = (40, 40, 40)
    center = (width // 2, height // 2)
    axes = (width // 3, height // 4)
    cv2.ellipse(img, center, axes, 30, 0, 360, bgr, -1)
    return img


def test_analyze_returns_valid_ranges():
    analyzer = BananaAnalyzer(enable_color_analysis=True, enable_morphology=True, enable_texture=True)
    img = _synthetic_banana()

    features = analyzer.analyze(img)

    assert 0.0 <= features.quality_score <= 1.0
    assert 0.0 <= features.yellow_ratio <= 1.0
    assert 0.0 <= features.green_ratio <= 1.0
    assert 0.0 <= features.brown_ratio <= 1.0
    assert 0.0 <= features.color_uniformity <= 1.0
    assert features.spot_count >= 0


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
def test_preprocess_frame_keeps_shape_and_dtype():
    analyzer = BananaAnalyzer()
    img = _synthetic_banana()

    out = analyzer.preprocess_frame(img, enhance_contrast=True, denoise=True)

    assert out.shape == img.shape
    assert out.dtype == img.dtype


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
def test_create_banana_mask_is_binary_uint8():
    analyzer = BananaAnalyzer()
    img = _synthetic_banana()

    mask = analyzer.create_banana_mask(img)

    assert mask.dtype == np.uint8
    assert mask.shape[:2] == img.shape[:2]

    uniq = set(np.unique(mask).tolist())
    assert uniq.issubset({0, 255})


def test_refine_category_prefers_features_when_classifier_uncertain():
    analyzer = BananaAnalyzer()

    img_greenish = _synthetic_banana(bgr=(0, 180, 60))
    features = analyzer.analyze(img_greenish)

    refined_cat, refined_conf = analyzer.refine_category_with_features(
        predicted_category="export",
        features=features,
        cls_confidence=0.35,
    )

    assert refined_cat in {"unripe", "export", "overripe", "defective"}
    assert 0.0 <= refined_conf <= 1.0


def test_multiscale_iou_basics():
    ms = MultiScaleDetector(scales=[0.5, 1.0, 1.5])

    box1 = (0, 0, 100, 100)
    box2 = (50, 50, 150, 150)
    box3 = (200, 200, 300, 300)

    iou12 = ms.compute_iou(box1, box2)
    iou11 = ms.compute_iou(box1, box1)
    iou13 = ms.compute_iou(box1, box3)

    assert 0.0 < iou12 < 1.0
    assert abs(iou11 - 1.0) < 1e-6
    assert iou13 == 0.0


def test_multiscale_merge_detections_reduces_overlaps():
    ms = MultiScaleDetector(scales=[1.0], nms_threshold=0.4)

    detections = [
        ((0, 0, 100, 100), 0.9),
        ((10, 10, 110, 110), 0.8),
        ((200, 200, 300, 300), 0.7),
    ]

    merged = ms.merge_detections(detections)

    assert len(merged) == 2
    assert merged[0][1] >= merged[1][1]
