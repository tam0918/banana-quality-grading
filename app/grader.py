from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


@dataclass(frozen=True)
class GradeResult:
    category_key: str  # one of: unripe, export, overripe, defective, none
    label_vi: str
    label_en: str
    status_vi: str
    confidence: float  # 0..1
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    debug: Dict[str, float]


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
        data_yaml_path: Optional[str] = None,
        device: Union[int, str, None] = None,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        # model_path is the *classifier* weights (e.g., weights/best.pt)
        self.model_path = model_path
        self.detector_model_path = detector_model_path
        self.data_yaml_path = data_yaml_path
        self.device = device
        self.conf = conf
        self.iou = iou

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

        if self.data_yaml_path:
            auto = self._mapping_from_data_yaml(self.data_yaml_path)
            if auto:
                self.class_id_to_category_key = auto

        self._load_models()

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
            self._detector = YOLO(self.detector_model_path)
            self._classifier = YOLO(self.model_path)
        except Exception as e:
            self._detector = None
            self._classifier = None
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
        return None

    def grade(self, frame_bgr: np.ndarray) -> GradeResult:
        if self._detector is None or self._classifier is None:
            msg = self._load_error or "Model chưa sẵn sàng."
            return GradeResult(
                category_key="none",
                label_vi="Chưa có model YOLO (best.pt)",
                label_en="Missing YOLO model (best.pt)",
                status_vi=msg,
                confidence=0.0,
                bbox_xyxy=None,
                debug={"error": 1.0},
            )

        # 1) Detect banana bbox
        banana_id = self._find_banana_class_id()
        if banana_id is None:
            return GradeResult(
                category_key="none",
                label_vi="Detector không có class banana",
                label_en="Detector missing banana class",
                status_vi="Hãy dùng detector COCO (yolov8n.pt) hoặc model có class 'banana'",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"error": 1.0},
            )

        try:
            det = self._detector.predict(
                source=frame_bgr,
                verbose=False,
                conf=max(0.15, self.conf),
                iou=self.iou,
                device=self.device,
            )
        except Exception as e:
            return GradeResult(
                category_key="none",
                label_vi="Lỗi detect",
                label_en="Detection error",
                status_vi=str(e),
                confidence=0.0,
                bbox_xyxy=None,
                debug={"error": 1.0},
            )

        if not det:
            return GradeResult(
                category_key="none",
                label_vi="Không phát hiện chuối",
                label_en="No banana detected",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"detections": 0.0},
            )

        boxes = getattr(det[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return GradeResult(
                category_key="none",
                label_vi="Không phát hiện chuối",
                label_en="No banana detected",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"detections": 0.0},
            )

        best_box = None
        best_det_conf = -1.0
        for b in boxes:
            try:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
            except Exception:
                cls_id = int(b.cls)
                conf = float(b.conf)
            if cls_id != banana_id:
                continue
            if conf > best_det_conf:
                best_det_conf = conf
                best_box = b

        if best_box is None:
            return GradeResult(
                category_key="none",
                label_vi="Không phát hiện chuối",
                label_en="No banana detected",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"detections": float(len(boxes))},
            )

        xyxy = best_box.xyxy
        try:
            x1, y1, x2, y2 = [int(v) for v in xyxy[0].tolist()]
        except Exception:
            x1, y1, x2, y2 = [int(v) for v in xyxy]

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
            cls_res = self._classifier.predict(source=crop, verbose=False, device=self.device)
        except Exception as e:
            return GradeResult(
                category_key="none",
                label_vi="Lỗi classify",
                label_en="Classification error",
                status_vi=str(e),
                confidence=0.0,
                bbox_xyxy=(x1, y1, x2, y2),
                debug={"det_conf": float(best_det_conf)},
            )

        if not cls_res:
            return GradeResult(
                category_key="none",
                label_vi="Không phân loại được",
                label_en="No classification result",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=(x1, y1, x2, y2),
                debug={"det_conf": float(best_det_conf)},
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
                debug={"det_conf": float(best_det_conf)},
            )

        try:
            top1 = int(probs.top1)
            top1_conf = float(probs.top1conf)
        except Exception:
            # Fallback
            top1 = int(getattr(probs, "top1", 0))
            top1_conf = float(getattr(probs, "top1conf", 0.0))

        class_names = getattr(self._classifier, "names", {})
        pred_name = str(class_names.get(top1, str(top1)))

        # Map predicted class name -> category
        inferred = self._infer_category_from_class_name(pred_name)
        category_key = inferred or self.class_id_to_category_key.get(top1) or "export"

        label_vi, label_en, status_vi = self._category_info[category_key]
        confidence = float(min(1.0, max(0.0, top1_conf)))

        debug = {
            "det_conf": float(best_det_conf),
            "cls_conf": float(confidence),
            "cls_id": float(top1),
        }

        return GradeResult(
            category_key=category_key,
            label_vi=label_vi,
            label_en=label_en,
            status_vi=status_vi,
            confidence=confidence,
            bbox_xyxy=(x1, y1, x2, y2),
            debug=debug,
        )
