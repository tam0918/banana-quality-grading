from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import os

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

try:
    from app.banana_analyzer import BananaAnalyzer, BananaFeatures, MultiScaleDetector
except Exception:  # pragma: no cover
    BananaAnalyzer = None  # type: ignore
    BananaFeatures = None  # type: ignore
    MultiScaleDetector = None  # type: ignore


@dataclass(frozen=True)
class GradeResult:
    category_key: str  # one of: unripe, export, overripe, defective, none
    label_vi: str
    label_en: str
    status_vi: str
    confidence: float  # 0..1
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    debug: Dict[str, float]
    # Enhanced features from BananaAnalyzer
    quality_score: float = 0.0  # 0-1 overall quality
    color_features: Optional[Dict[str, float]] = None
    spot_count: int = 0
    refined: bool = False  # True if refined by feature analysis


@dataclass(frozen=True)
class FrameGrade:
    """Result for a whole frame.

    - items: per-banana results (multi-bbox)
    - overall: a single summary result for the UI panel / backward compat
    """

    overall: GradeResult
    items: List[GradeResult]


class BananaGrader:
    """YOLOv8-based banana grading using a 2-stage pipeline.

    Why 2-stage?
    - Many public datasets (incl. Kaggle) are *classification* (folder-per-class) and have no bbox.
    - UI requires bbox overlay.

    Pipeline:
    1) Detector (COCO) finds banana bbox.
    2) Classifier (trained on ripeness dataset) predicts ripeness class for the cropped banana.

    This class then maps predicted class name -> the 4 required actionable categories.
    """

    def __init__(
        self,
        model_path: str,
        detector_model_path: str = "yolov8n.pt",
        detector_backend: Optional[str] = None,
        haar_cascade_path: Optional[str] = None,
        data_yaml_path: Optional[str] = None,
        device: Union[int, str, None] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        # Enhanced analysis options (based on research paper)
        enable_enhanced_analysis: bool = True,
        enable_multi_scale: bool = False,
        enable_preprocessing: bool = True,
        enable_feature_refinement: bool = True,
    ):
        # model_path is the *classifier* weights (e.g., weights/best.pt)
        self.model_path = model_path
        self.detector_model_path = detector_model_path
        env = os.environ
        self.detector_backend = (detector_backend or env.get("BANANA_DETECTOR_BACKEND", "yolo")).strip().lower()
        self.haar_cascade_path = (
            haar_cascade_path
            or env.get("BANANA_HAAR_PATH", "").strip()
            or "haarbanana.xml"
        )
        self.data_yaml_path = data_yaml_path
        # Device handling
        # - Ultralytics accepts: "cpu", "0", "0,1" ...
        # - If not provided, we default to GPU (cuda:0) when available.
        # - Optional override via env BANANA_DEVICE (e.g., "cpu" or "0")
        env_device = ("" if device is not None else __import__("os").environ.get("BANANA_DEVICE", "")).strip()
        self.device = self._pick_device(device if device is not None else (env_device or "auto"))
        self.conf = conf
        self.iou = iou

        # How many bananas to analyze per frame when multi-detecting.
        # Keep small to preserve real-time performance on CPU.
        try:
            self.max_fruits = max(1, int(env.get("BANANA_MAX_FRUITS", "6")))
        except Exception:
            self.max_fruits = 6

        # Inference performance knobs (safe defaults).
        # - imgsz: smaller = faster but may reduce detection accuracy.
        # - half: GPU-only optimization.
        try:
            self.det_imgsz = max(128, int(env.get("BANANA_DET_IMGSZ", "640")))
        except Exception:
            self.det_imgsz = 640
        try:
            self.cls_imgsz = max(128, int(env.get("BANANA_CLS_IMGSZ", "416")))
        except Exception:
            self.cls_imgsz = 416

        self.half = (env.get("BANANA_HALF", "").strip().lower() in {"1", "true", "yes", "y"})

        # Analyzer throttling (to keep UI smooth on CPU).
        # - all: run analyzer on every crop
        # - defective: only when predicted defective
        # - uncertain: only when cls confidence < BANANA_ANALYZE_UNCERTAIN_THRESH
        # - defective_or_uncertain: union of the above
        self.analyze_policy = (env.get("BANANA_ANALYZE_POLICY", "all").strip().lower() or "all")
        try:
            self.analyze_uncertain_thresh = float(env.get("BANANA_ANALYZE_UNCERTAIN_THRESH", "0"))
        except Exception:
            self.analyze_uncertain_thresh = 0.0
        try:
            self.analyze_every = max(1, int(env.get("BANANA_ANALYZE_EVERY", "1")))
        except Exception:
            self.analyze_every = 1
        self._frame_index = 0

        # IMPORTANT:
        # - If you already know your dataset's class order, you can hardcode it here.
        # - Otherwise, pass data_yaml_path="datasets/data.yaml" to auto-map by class names.
        self.class_id_to_category_key = self._default_mapping()

        self._category_info = {
            "unripe": ("Chuối Xanh", "Unripe", "Chưa thu hoạch"),
            "export": ("Chín Vừa/Xuất Khẩu", "Just Ripe / Export", "Bảo quản & Ship hàng"),
            "overripe": ("Quá Chín", "Overripe", "Cần bán/Ăn ngay"),
            "defective": ("Bị bệnh/Hỏng", "Defective", "Loại bỏ"),
        }

        self._detector = None
        self._classifier = None
        self._load_error: Optional[str] = None

        self._haar = None

        # Temporal stabilization (useful when you don't have a custom detector dataset).
        # Holds the last successful bbox for N subsequent frames if detector temporarily misses.
        # Disable by setting BANANA_BBOX_HOLD=0.
        try:
            self._bbox_hold_frames = max(0, int(env.get("BANANA_BBOX_HOLD", "5")))
        except Exception:
            self._bbox_hold_frames = 5
        self._bbox_hold_remaining = 0
        self._last_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None

        if self.data_yaml_path:
            auto = self._mapping_from_data_yaml(self.data_yaml_path)
            if auto:
                self.class_id_to_category_key = auto

        # Enhanced analysis settings (based on research paper methodology)
        self.enable_enhanced_analysis = enable_enhanced_analysis
        self.enable_multi_scale = enable_multi_scale
        self.enable_preprocessing = enable_preprocessing
        self.enable_feature_refinement = enable_feature_refinement
        
        # Initialize BananaAnalyzer for advanced feature extraction
        self._analyzer: Optional[BananaAnalyzer] = None
        self._multi_scale_detector: Optional[MultiScaleDetector] = None
        
        if BananaAnalyzer is not None and self.enable_enhanced_analysis:
            self._analyzer = BananaAnalyzer(
                enable_color_analysis=True,
                enable_morphology=True,
                enable_texture=True,
            )
            if self.enable_multi_scale and MultiScaleDetector is not None:
                self._multi_scale_detector = MultiScaleDetector(
                    scales=[0.75, 1.0, 1.25],
                    nms_threshold=0.4,
                )

        self._load_models()

    def _update_last_bbox(self, bbox_xyxy: Tuple[int, int, int, int]) -> None:
        if self._bbox_hold_frames <= 0:
            return
        self._last_bbox_xyxy = bbox_xyxy
        self._bbox_hold_remaining = int(self._bbox_hold_frames)

    def _try_use_held_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self._bbox_hold_frames <= 0:
            return None
        if self._last_bbox_xyxy is None:
            return None
        if self._bbox_hold_remaining <= 0:
            return None

        self._bbox_hold_remaining -= 1

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self._last_bbox_xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _load_haar(self) -> Optional[object]:
        if cv2 is None:
            self._load_error = "Chưa cài OpenCV (opencv-python)."
            return None
        try:
            cascade = cv2.CascadeClassifier(self.haar_cascade_path)
            # OpenCV returns empty classifier if load failed
            if cascade.empty():
                self._load_error = f"Không load được Haar Cascade: {self.haar_cascade_path}"
                return None
            return cascade
        except Exception as e:
            self._load_error = f"Không load được Haar Cascade: {type(e).__name__}: {e}"
            return None

    @staticmethod
    def _pick_device(requested: Union[int, str, None]) -> Union[int, str, None]:
        if requested is None:
            return None
        if isinstance(requested, int):
            return requested
        req = str(requested).strip()
        if not req:
            return None

        if req.lower() != "auto":
            return req

        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _default_mapping() -> Dict[int, str]:
        # Typical Roboflow ripeness datasets:
        # 0: fresh/green, 1: ripe/yellow, 2: overripe, 3: rotten
        return {
            0: "unripe",
            1: "export",
            2: "overripe",
            3: "defective",
        }

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else " " for ch in name).strip()

    def _infer_category_from_class_name(self, name: str) -> Optional[str]:
        n = self._normalize_name(name)
        tokens = set(n.split())

        # Defective/rotten signals
        if tokens.intersection({"rotten", "rot", "bad", "mold", "mould", "disease", "defect", "defective", "spoiled"}):
            return "defective"

        # Overripe signals
        if tokens.intersection({"overripe", "over", "brown", "spotted", "spot", "bruise"}):
            return "overripe"

        # Unripe/green/fresh signals
        if tokens.intersection({"unripe", "green", "fresh", "raw"}):
            return "unripe"

        # Ripe/yellow signals
        if tokens.intersection({"ripe", "yellow", "mature"}):
            return "export"

        return None

    def _mapping_from_data_yaml(self, path: str) -> Optional[Dict[int, str]]:
        if yaml is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            return None

        names: Optional[Union[List[str], Dict[Union[int, str], str]]] = data.get("names") if isinstance(data, dict) else None
        if not names:
            return None

        # YOLO data.yaml can store names as list or dict
        if isinstance(names, list):
            ordered = list(names)
        elif isinstance(names, dict):
            # keys could be "0".."n" or ints
            ordered = [v for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        else:
            return None

        mapping: Dict[int, str] = {}
        for idx, class_name in enumerate(ordered):
            cat = self._infer_category_from_class_name(str(class_name))
            if cat is not None:
                mapping[int(idx)] = cat

        # Only accept mapping if it covers at least 2 classes; otherwise keep default.
        return mapping if len(mapping) >= 2 else None

    def _load_models(self) -> None:
        if YOLO is None:
            self._load_error = "Chưa cài ultralytics. Hãy pip install ultralytics."
            return
        try:
            # Detector can be YOLO or Haar Cascade.
            if self.detector_backend == "haar":
                self._detector = None
                self._haar = self._load_haar()
            else:
                self._detector = YOLO(self.detector_model_path)
                self._haar = None
            self._classifier = YOLO(self.model_path)

            # Speed up inference: fuse Conv+BN layers when supported.
            try:
                if self._detector is not None:
                    self._detector.fuse()
            except Exception:
                pass
            try:
                if self._classifier is not None:
                    self._classifier.fuse()
            except Exception:
                pass

            # CPU thread tuning (often helps smoothness on weaker CPUs).
            try:
                if str(self.device).lower() == "cpu":
                    import torch
                    req = os.environ.get("BANANA_TORCH_THREADS", "").strip()
                    if req:
                        n = max(1, int(req))
                    else:
                        n = max(1, min(4, int(os.cpu_count() or 4)))
                    torch.set_num_threads(n)
                    try:
                        torch.set_num_interop_threads(1)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            self._detector = None
            self._classifier = None
            self._haar = None
            self._load_error = (
                f"Không load được model. detector={self.detector_model_path}, classifier={self.model_path}. "
                f"Lỗi: {type(e).__name__}: {e}"
            )

    def _find_banana_class_id(self) -> Optional[int]:
        if self._detector is None:
            return None
        names = getattr(self._detector, "names", None)
        if not isinstance(names, dict):
            return None
        for k, v in names.items():
            if str(v).strip().lower() == "banana":
                return int(k)
        # If user trains a custom single-class detector and the class name isn't "banana",
        # accept it to avoid a hard failure.
        try:
            if len(names) == 1:
                return int(next(iter(names.keys())))
        except Exception:
            pass
        return None

    def _haar_detect_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self._haar is None or cv2 is None:
            return None
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

        try:
            rects = self._haar.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                flags=getattr(cv2, "CASCADE_SCALE_IMAGE", 0),
            )
        except Exception:
            return None

        if rects is None or len(rects) == 0:
            return None

        # Pick the largest rectangle (most likely the banana)
        best = None
        best_area = -1
        for (x, y, w, h) in rects:
            area = int(w) * int(h)
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(x + w), int(y + h))
        return best

    def _haar_detect_bboxes(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._haar is None or cv2 is None:
            return []
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return []

        try:
            rects = self._haar.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                flags=getattr(cv2, "CASCADE_SCALE_IMAGE", 0),
            )
        except Exception:
            return []

        if rects is None or len(rects) == 0:
            return []

        # Sort by area (desc) and take top-N.
        candidates: List[Tuple[int, int, int, int, int]] = []
        for (x, y, w, h) in rects:
            area = int(w) * int(h)
            candidates.append((area, int(x), int(y), int(x + w), int(y + h)))
        candidates.sort(key=lambda t: t[0], reverse=True)
        return [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in candidates[: self.max_fruits]]

    @staticmethod
    def _severity_rank(category_key: str) -> int:
        # Higher = more severe (used for overall aggregation)
        if category_key == "defective":
            return 4
        if category_key == "overripe":
            return 3
        if category_key == "export":
            return 2
        if category_key == "unripe":
            return 1
        return 0

    def _aggregate_overall(self, items: List[GradeResult]) -> GradeResult:
        if not items:
            return GradeResult(
                category_key="none",
                label_vi="Không phát hiện chuối",
                label_en="No banana detected",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"detections": 0.0},
            )

        # Pick the single most severe result; tie-break by confidence.
        best = max(items, key=lambda r: (self._severity_rank(r.category_key), float(r.confidence)))

        # Annotate summary with count (keeps UI minimal: reuse existing status line)
        debug = dict(best.debug or {})
        debug["detections"] = float(len(items))
        return GradeResult(
            category_key=best.category_key,
            label_vi=best.label_vi,
            label_en=best.label_en,
            status_vi=f"{best.status_vi} | Số quả: {len(items)}",
            confidence=best.confidence,
            bbox_xyxy=best.bbox_xyxy,
            debug=debug,
            quality_score=best.quality_score,
            color_features=best.color_features,
            spot_count=best.spot_count,
            refined=best.refined,
        )

    def _grade_bbox(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        det_conf: float,
        bbox_held: bool,
    ) -> GradeResult:
        x1, y1, x2, y2 = bbox_xyxy

        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return GradeResult(
                category_key="none",
                label_vi="BBox không hợp lệ",
                label_en="Invalid bbox",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"error": 1.0},
            )

        crop = frame_bgr[y1:y2, x1:x2]

        # 2) Classify crop
        try:
            cls_res = self._classifier.predict(
                source=crop,
                verbose=False,
                device=self.device,
                imgsz=int(self.cls_imgsz),
                half=bool(self.half),
            )
        except Exception as e:
            return GradeResult(
                category_key="none",
                label_vi="Lỗi classify",
                label_en="Classification error",
                status_vi=str(e),
                confidence=0.0,
                bbox_xyxy=(x1, y1, x2, y2),
                debug={"det_conf": float(det_conf)},
            )

        if not cls_res:
            return GradeResult(
                category_key="none",
                label_vi="Không phân loại được",
                label_en="No classification result",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=(x1, y1, x2, y2),
                debug={"det_conf": float(det_conf)},
            )

        r0 = cls_res[0]
        probs = getattr(r0, "probs", None)
        if probs is None:
            return GradeResult(
                category_key="none",
                label_vi="Không phân loại được",
                label_en="No classification result",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=(x1, y1, x2, y2),
                debug={"det_conf": float(det_conf)},
            )

        try:
            top1 = int(probs.top1)
            top1_conf = float(probs.top1conf)
        except Exception:
            top1 = int(getattr(probs, "top1", 0))
            top1_conf = float(getattr(probs, "top1conf", 0.0))

        class_names = getattr(self._classifier, "names", {})
        pred_name = str(class_names.get(top1, str(top1)))

        inferred = self._infer_category_from_class_name(pred_name)
        category_key = inferred or self.class_id_to_category_key.get(top1) or "export"

        label_vi, label_en, status_vi = self._category_info[category_key]
        confidence = float(min(1.0, max(0.0, top1_conf)))

        debug: Dict[str, float] = {
            "det_conf": float(det_conf),
            "cls_conf": float(confidence),
            "cls_id": float(top1),
        }

        if bbox_held:
            debug["bbox_held"] = 1.0

        # ============================================================
        # Enhanced Analysis
        # ============================================================
        quality_score = 0.0
        color_features = None
        spot_count = 0
        refined = False

        should_analyze = False
        if self._analyzer is not None and self.enable_enhanced_analysis:
            # Frame-based throttling
            if self.analyze_every <= 1 or (self._frame_index % self.analyze_every) == 0:
                should_analyze = True

            # Policy-based gating
            pol = self.analyze_policy
            if pol == "never":
                should_analyze = False
            elif pol == "all":
                pass
            elif pol == "defective":
                should_analyze = (category_key == "defective")
            elif pol == "uncertain":
                should_analyze = (float(confidence) < float(self.analyze_uncertain_thresh))
            elif pol == "defective_or_uncertain":
                should_analyze = (category_key == "defective") or (float(confidence) < float(self.analyze_uncertain_thresh))

        if should_analyze:
            try:
                if self.enable_preprocessing:
                    processed_crop = self._analyzer.preprocess_frame(
                        crop, enhance_contrast=True, denoise=True
                    )
                else:
                    processed_crop = crop

                features = self._analyzer.analyze(processed_crop)

                quality_score = features.quality_score
                spot_count = features.spot_count
                color_features = {
                    "yellow_ratio": features.yellow_ratio,
                    "green_ratio": features.green_ratio,
                    "brown_ratio": features.brown_ratio,
                    "color_uniformity": features.color_uniformity,
                    "solidity": features.solidity,
                    "texture_variance": features.texture_variance,
                }

                debug["quality_score"] = float(quality_score)
                debug["yellow_ratio"] = float(features.yellow_ratio)
                debug["green_ratio"] = float(features.green_ratio)
                debug["brown_ratio"] = float(features.brown_ratio)
                debug["spot_count"] = float(spot_count)

                if self.enable_feature_refinement:
                    refined_category, refined_conf = self._analyzer.refine_category_with_features(
                        category_key, features, confidence
                    )

                    if refined_category != category_key:
                        category_key = refined_category
                        label_vi, label_en, status_vi = self._category_info[category_key]
                        refined = True
                        debug["refined"] = 1.0

                    confidence = float(refined_conf)
                    debug["ensemble_conf"] = float(refined_conf)

            except Exception as e:
                debug["analyzer_error"] = 1.0
                debug["analyzer_error_msg"] = 1.0

        return GradeResult(
            category_key=category_key,
            label_vi=label_vi,
            label_en=label_en,
            status_vi=status_vi,
            confidence=float(confidence),
            bbox_xyxy=(x1, y1, x2, y2),
            debug=debug,
            quality_score=float(quality_score),
            color_features=color_features,
            spot_count=int(spot_count),
            refined=bool(refined),
        )

    def grade_frame(self, frame_bgr: np.ndarray) -> FrameGrade:
        """Grade a frame and return per-banana results + an overall summary."""

        # Used by analyzer throttling.
        self._frame_index += 1

        if (self.detector_backend == "haar" and self._haar is None) or self._classifier is None:
            msg = self._load_error or "Model chưa sẵn sàng."
            overall = GradeResult(
                category_key="none",
                label_vi="Chưa có model YOLO (best.pt)",
                label_en="Missing YOLO model (best.pt)",
                status_vi=msg,
                confidence=0.0,
                bbox_xyxy=None,
                debug={"error": 1.0},
            )
            return FrameGrade(overall=overall, items=[])

        if self.detector_backend != "haar" and self._detector is None:
            msg = self._load_error or "Model chưa sẵn sàng."
            overall = GradeResult(
                category_key="none",
                label_vi="Chưa có model YOLO (best.pt)",
                label_en="Missing YOLO model (best.pt)",
                status_vi=msg,
                confidence=0.0,
                bbox_xyxy=None,
                debug={"error": 1.0},
            )
            return FrameGrade(overall=overall, items=[])

        detections: List[Tuple[Tuple[int, int, int, int], float, bool]] = []

        if self.detector_backend == "haar":
            bboxes = self._haar_detect_bboxes(frame_bgr)
            if not bboxes:
                held = self._try_use_held_bbox(frame_bgr)
                if held is not None:
                    detections = [(held, 0.01, True)]
                else:
                    return FrameGrade(overall=self._aggregate_overall([]), items=[])
            else:
                # Haar has no confidence; treat as 1.0 and update hold bbox from the largest.
                self._update_last_bbox(bboxes[0])
                detections = [(bb, 1.0, False) for bb in bboxes]

        else:
            banana_id = self._find_banana_class_id()
            if banana_id is None:
                overall = GradeResult(
                    category_key="none",
                    label_vi="Detector không có class banana",
                    label_en="Detector missing banana class",
                    status_vi="Hãy dùng detector COCO (yolov8n.pt) hoặc model có class 'banana'",
                    confidence=0.0,
                    bbox_xyxy=None,
                    debug={"error": 1.0},
                )
                return FrameGrade(overall=overall, items=[])

            try:
                det = self._detector.predict(
                    source=frame_bgr,
                    verbose=False,
                    conf=max(0.15, self.conf),
                    iou=self.iou,
                    device=self.device,
                    imgsz=int(self.det_imgsz),
                    classes=[int(banana_id)],
                    half=bool(self.half),
                )
            except Exception as e:
                overall = GradeResult(
                    category_key="none",
                    label_vi="Lỗi detect",
                    label_en="Detection error",
                    status_vi=str(e),
                    confidence=0.0,
                    bbox_xyxy=None,
                    debug={"error": 1.0},
                )
                return FrameGrade(overall=overall, items=[])

            boxes = getattr(det[0], "boxes", None) if det else None
            if boxes is None or len(boxes) == 0:
                held = self._try_use_held_bbox(frame_bgr)
                if held is not None:
                    detections = [(held, 0.01, True)]
                else:
                    return FrameGrade(overall=self._aggregate_overall([]), items=[])
            else:
                candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
                for b in boxes:
                    try:
                        cls_id = int(b.cls.item())
                        conf = float(b.conf.item())
                    except Exception:
                        cls_id = int(b.cls)
                        conf = float(b.conf)
                    if cls_id != banana_id:
                        continue
                    xyxy = b.xyxy
                    try:
                        x1, y1, x2, y2 = [int(v) for v in xyxy[0].tolist()]
                    except Exception:
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                    candidates.append((conf, (x1, y1, x2, y2)))

                candidates.sort(key=lambda t: t[0], reverse=True)
                if not candidates:
                    held = self._try_use_held_bbox(frame_bgr)
                    if held is not None:
                        detections = [(held, 0.01, True)]
                    else:
                        return FrameGrade(overall=self._aggregate_overall([]), items=[])
                else:
                    # Update bbox-hold with the top detection.
                    self._update_last_bbox(candidates[0][1])
                    detections = [(bb, float(conf), False) for conf, bb in candidates[: self.max_fruits]]

        items: List[GradeResult] = []
        for bbox, det_conf, held in detections:
            items.append(self._grade_bbox(frame_bgr, bbox, det_conf=det_conf, bbox_held=held))

        overall = self._aggregate_overall(items)
        return FrameGrade(overall=overall, items=items)

    def grade(self, frame_bgr: np.ndarray) -> GradeResult:
        # Backward-compatible single-result API.
        return self.grade_frame(frame_bgr).overall
