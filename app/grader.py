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
    """YOLOv8-based banana grading (Object Detection).

    The model returns:
    - bbox [x1,y1,x2,y2]
    - class id
    - confidence

    This class maps class ids to the 4 required actionable categories.
    """

    def __init__(
        self,
        model_path: str,
        data_yaml_path: Optional[str] = None,
        device: Union[int, str, None] = None,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        self.model_path = model_path
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

        self._model = None
        self._load_error: Optional[str] = None

        if self.data_yaml_path:
            auto = self._mapping_from_data_yaml(self.data_yaml_path)
            if auto:
                self.class_id_to_category_key = auto

        self._load_model()

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

    def _load_model(self) -> None:
        if YOLO is None:
            self._load_error = "Chưa cài ultralytics. Hãy pip install ultralytics."
            return
        try:
            self._model = YOLO(self.model_path)
        except Exception as e:
            self._model = None
            self._load_error = (
                f"Không load được model weights: {self.model_path}. "
                f"Lỗi: {type(e).__name__}: {e}"
            )

    def grade(self, frame_bgr: np.ndarray) -> GradeResult:
        if self._model is None:
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

        # Inference
        # Ultralytics accepts BGR numpy arrays directly.
        try:
            results = self._model.predict(
                source=frame_bgr,
                verbose=False,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
            )
        except Exception as e:
            return GradeResult(
                category_key="none",
                label_vi="Lỗi inference",
                label_en="Inference error",
                status_vi=str(e),
                confidence=0.0,
                bbox_xyxy=None,
                debug={"error": 1.0},
            )

        if not results:
            return GradeResult(
                category_key="none",
                label_vi="Không phát hiện chuối",
                label_en="No banana detected",
                status_vi="—",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"detections": 0.0},
            )

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
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

        best = None
        best_conf = -1.0

        # Each box has: xyxy, cls, conf
        for b in boxes:
            try:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
            except Exception:
                # Fallback for older ultralytics tensor shapes
                cls_id = int(b.cls)
                conf = float(b.conf)

            if cls_id not in self.class_id_to_category_key:
                continue
            if conf > best_conf:
                best_conf = conf
                best = b

        if best is None:
            return GradeResult(
                category_key="none",
                label_vi="Không đúng class dataset",
                label_en="Unknown dataset classes",
                status_vi="Hãy chỉnh mapping class_id_to_category_key",
                confidence=0.0,
                bbox_xyxy=None,
                debug={"detections": float(len(boxes))},
            )

        # Extract bbox
        xyxy = best.xyxy
        try:
            x1, y1, x2, y2 = [int(v) for v in xyxy[0].tolist()]
        except Exception:
            x1, y1, x2, y2 = [int(v) for v in xyxy]

        category_key = self.class_id_to_category_key[int(best.cls.item())]
        label_vi, label_en, status_vi = self._category_info[category_key]

        debug = {
            "detections": float(len(boxes)),
            "best_conf": float(best_conf),
            "class_id": float(int(best.cls.item())),
        }

        return GradeResult(
            category_key=category_key,
            label_vi=label_vi,
            label_en=label_en,
            status_vi=status_vi,
            confidence=float(best_conf),
            bbox_xyxy=(x1, y1, x2, y2),
            debug=debug,
        )
