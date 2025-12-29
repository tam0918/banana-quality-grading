"""Enhanced Banana Analyzer Module.

Based on: "Comparative Analysis of Banana Detection Models: Deep Learning and Darknet Algorithm"
http://ijeces.ferit.hr/index.php/ijeces/article/view/3043

This module implements advanced image processing techniques for improved banana detection
and quality assessment:
- Color-based preprocessing (HSV/LAB color spaces)
- Morphological feature extraction
- Multi-scale detection support
- Quality metrics computation
- Ensemble confidence scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class BananaFeatures:
    """Extracted features for banana quality assessment."""
    # Color features
    yellow_ratio: float  # Ratio of yellow pixels
    green_ratio: float   # Ratio of green pixels
    brown_ratio: float   # Ratio of brown/spot pixels
    color_uniformity: float  # Color distribution uniformity
    
    # Morphological features
    area: float  # Banana area in pixels
    perimeter: float  # Banana perimeter
    aspect_ratio: float  # Width/Height ratio
    curvature: float  # Banana curvature estimation
    solidity: float  # Area / Convex Hull Area
    
    # Texture features
    texture_variance: float  # Texture roughness indicator
    spot_count: int  # Number of dark spots detected
    
    # Quality score
    quality_score: float  # Overall computed quality (0-1)


class BananaAnalyzer:
    """Advanced banana image analysis based on research paper methodology."""
    
    # HSV ranges for banana color classification (optimized for banana detection)
    # Yellow banana (ripe)
    YELLOW_HSV_LOW = np.array([15, 80, 80])
    YELLOW_HSV_HIGH = np.array([35, 255, 255])
    
    # Green banana (unripe)
    GREEN_HSV_LOW = np.array([35, 40, 40])
    GREEN_HSV_HIGH = np.array([85, 255, 255])
    
    # Brown spots (overripe/defective)
    BROWN_HSV_LOW = np.array([5, 50, 20])
    BROWN_HSV_HIGH = np.array([20, 200, 150])
    
    # Black spots (defective)
    BLACK_HSV_LOW = np.array([0, 0, 0])
    BLACK_HSV_HIGH = np.array([180, 255, 50])
    
    def __init__(
        self,
        enable_color_analysis: bool = True,
        enable_morphology: bool = True,
        enable_texture: bool = True,
        spot_detection_threshold: float = 0.02,
    ):
        """Initialize the analyzer.
        
        Args:
            enable_color_analysis: Enable HSV/LAB color feature extraction
            enable_morphology: Enable morphological feature extraction
            enable_texture: Enable texture analysis
            spot_detection_threshold: Min ratio to detect spots as significant
        """
        self.enable_color_analysis = enable_color_analysis
        self.enable_morphology = enable_morphology
        self.enable_texture = enable_texture
        self.spot_detection_threshold = spot_detection_threshold
        
    def preprocess_frame(
        self,
        frame_bgr: np.ndarray,
        enhance_contrast: bool = True,
        denoise: bool = True,
    ) -> np.ndarray:
        """Preprocess frame for better banana detection.
        
        Based on paper methodology: contrast enhancement and noise reduction
        improve detection accuracy by 5-10%.
        
        Args:
            frame_bgr: Input BGR frame
            enhance_contrast: Apply CLAHE contrast enhancement
            denoise: Apply Gaussian denoising
            
        Returns:
            Preprocessed frame
        """
        if cv2 is None:
            return frame_bgr
            
        result = frame_bgr.copy()
        
        # Denoise using bilateral filter (edge-preserving)
        if denoise:
            result = cv2.bilateralFilter(result, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Convert to LAB color space for better contrast enhancement
        if enhance_contrast:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            
            # Merge channels back
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def create_banana_mask(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Create a binary mask highlighting banana regions using color segmentation.
        
        Args:
            frame_bgr: Input BGR frame
            bbox_xyxy: Optional bounding box to focus on
            
        Returns:
            Binary mask (255 for banana regions, 0 otherwise)
        """
        if cv2 is None:
            h, w = frame_bgr.shape[:2]
            return np.ones((h, w), dtype=np.uint8) * 255
        
        # Crop to bbox if provided
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy
            frame_bgr = frame_bgr[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Create masks for different banana colors
        yellow_mask = cv2.inRange(hsv, self.YELLOW_HSV_LOW, self.YELLOW_HSV_HIGH)
        green_mask = cv2.inRange(hsv, self.GREEN_HSV_LOW, self.GREEN_HSV_HIGH)
        
        # Combine masks (banana can be yellow OR green)
        banana_mask = cv2.bitwise_or(yellow_mask, green_mask)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_CLOSE, kernel)
        banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_OPEN, kernel)
        
        return banana_mask
    
    def extract_color_features(
        self,
        crop_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract color-based features from banana region.
        
        Based on paper: Color ratio analysis in HSV space provides reliable
        ripeness classification.
        
        Args:
            crop_bgr: Cropped banana image
            mask: Optional mask to focus analysis
            
        Returns:
            Dictionary of color features
        """
        if cv2 is None:
            return {
                "yellow_ratio": 0.5,
                "green_ratio": 0.2,
                "brown_ratio": 0.1,
                "black_ratio": 0.0,
                "color_uniformity": 0.8,
            }
        
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        
        # Calculate total pixels to consider
        if mask is not None:
            total_pixels = cv2.countNonZero(mask)
            if total_pixels == 0:
                total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
        else:
            total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
            mask = np.ones(crop_bgr.shape[:2], dtype=np.uint8) * 255
        
        # Create color masks
        yellow_mask = cv2.inRange(hsv, self.YELLOW_HSV_LOW, self.YELLOW_HSV_HIGH)
        green_mask = cv2.inRange(hsv, self.GREEN_HSV_LOW, self.GREEN_HSV_HIGH)
        brown_mask = cv2.inRange(hsv, self.BROWN_HSV_LOW, self.BROWN_HSV_HIGH)
        black_mask = cv2.inRange(hsv, self.BLACK_HSV_LOW, self.BLACK_HSV_HIGH)
        
        # Apply ROI mask
        yellow_mask = cv2.bitwise_and(yellow_mask, mask)
        green_mask = cv2.bitwise_and(green_mask, mask)
        brown_mask = cv2.bitwise_and(brown_mask, mask)
        black_mask = cv2.bitwise_and(black_mask, mask)
        
        # Calculate ratios
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
        green_ratio = cv2.countNonZero(green_mask) / total_pixels
        brown_ratio = cv2.countNonZero(brown_mask) / total_pixels
        black_ratio = cv2.countNonZero(black_mask) / total_pixels
        
        # Color uniformity using LAB color space standard deviation
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_std = np.std(lab[:, :, 0])
        a_std = np.std(lab[:, :, 1])
        b_std = np.std(lab[:, :, 2])
        # Normalize to 0-1 (lower std = more uniform)
        color_uniformity = max(0.0, 1.0 - (l_std + a_std + b_std) / 300.0)
        
        return {
            "yellow_ratio": yellow_ratio,
            "green_ratio": green_ratio,
            "brown_ratio": brown_ratio,
            "black_ratio": black_ratio,
            "color_uniformity": color_uniformity,
        }
    
    def extract_morphological_features(
        self,
        crop_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract morphological features from banana region.
        
        Based on paper: Shape features help distinguish healthy bananas
        from damaged ones.
        
        Args:
            crop_bgr: Cropped banana image
            mask: Binary mask of banana region
            
        Returns:
            Dictionary of morphological features
        """
        if cv2 is None:
            h, w = crop_bgr.shape[:2]
            return {
                "area": float(h * w),
                "perimeter": float(2 * (h + w)),
                "aspect_ratio": w / max(h, 1),
                "curvature": 0.5,
                "solidity": 0.9,
                "extent": 0.8,
            }
        
        # Create mask if not provided
        if mask is None:
            mask = self.create_banana_mask(crop_bgr)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            h, w = crop_bgr.shape[:2]
            return {
                "area": float(h * w),
                "perimeter": float(2 * (h + w)),
                "aspect_ratio": w / max(h, 1),
                "curvature": 0.5,
                "solidity": 0.9,
                "extent": 0.8,
            }
        
        # Get largest contour (main banana)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate features
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / max(float(h), 1.0)
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1.0)
        
        # Extent (area / bounding rect area)
        rect_area = w * h
        extent = area / max(rect_area, 1.0)
        
        # Curvature estimation using ellipse fitting
        curvature = 0.5
        if len(main_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(main_contour)
                (center, (width, height), angle) = ellipse
                # Typical banana has aspect ratio 2.5-4.0
                ellipse_ratio = max(width, height) / max(min(width, height), 1.0)
                curvature = min(1.0, max(0.0, (ellipse_ratio - 1.0) / 5.0))
            except Exception:
                pass
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "aspect_ratio": aspect_ratio,
            "curvature": curvature,
            "solidity": solidity,
            "extent": extent,
        }
    
    def detect_spots(
        self,
        crop_bgr: np.ndarray,
        min_spot_area: int = 10,
    ) -> Tuple[int, float]:
        """Detect dark spots/blemishes on banana surface.
        
        Based on paper: Spot detection is crucial for defect classification.
        
        Args:
            crop_bgr: Cropped banana image
            min_spot_area: Minimum pixel area to count as a spot
            
        Returns:
            Tuple of (spot_count, spot_area_ratio)
        """
        if cv2 is None:
            return 0, 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to detect dark regions
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Also detect brown/black spots using HSV
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        brown_mask = cv2.inRange(hsv, self.BROWN_HSV_LOW, self.BROWN_HSV_HIGH)
        black_mask = cv2.inRange(hsv, self.BLACK_HSV_LOW, self.BLACK_HSV_HIGH)
        
        # Combine spot masks
        spot_mask = cv2.bitwise_or(thresh, brown_mask)
        spot_mask = cv2.bitwise_or(spot_mask, black_mask)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)
        
        # Find spot contours
        contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant spots
        spot_count = 0
        total_spot_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_spot_area:
                spot_count += 1
                total_spot_area += area
        
        total_area = crop_bgr.shape[0] * crop_bgr.shape[1]
        spot_ratio = total_spot_area / max(total_area, 1)
        
        return spot_count, spot_ratio
    
    def compute_texture_variance(
        self,
        crop_bgr: np.ndarray,
    ) -> float:
        """Compute texture variance as roughness indicator.
        
        Based on paper: Texture features help identify surface damage.
        
        Args:
            crop_bgr: Cropped banana image
            
        Returns:
            Texture variance value (higher = rougher surface)
        """
        if cv2 is None:
            return 0.1
        
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance (measure of texture/edges)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to reasonable range (typical values 0-1000)
        normalized = min(1.0, variance / 1000.0)
        
        return normalized
    
    def analyze(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    ) -> BananaFeatures:
        """Perform full banana analysis.
        
        Args:
            frame_bgr: Input BGR frame
            bbox_xyxy: Bounding box (x1, y1, x2, y2) of detected banana
            
        Returns:
            BananaFeatures dataclass with all extracted features
        """
        # Crop to bbox if provided
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy
            h, w = frame_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame_bgr[y1:y2, x1:x2]
        else:
            crop = frame_bgr
        
        if crop.size == 0:
            return self._default_features()
        
        # Create banana mask for targeted analysis
        mask = self.create_banana_mask(crop) if self.enable_color_analysis else None
        
        # Extract color features
        if self.enable_color_analysis:
            color_features = self.extract_color_features(crop, mask)
        else:
            color_features = {
                "yellow_ratio": 0.5,
                "green_ratio": 0.2,
                "brown_ratio": 0.1,
                "black_ratio": 0.0,
                "color_uniformity": 0.8,
            }
        
        # Extract morphological features
        if self.enable_morphology:
            morph_features = self.extract_morphological_features(crop, mask)
        else:
            h, w = crop.shape[:2]
            morph_features = {
                "area": float(h * w),
                "perimeter": float(2 * (h + w)),
                "aspect_ratio": w / max(h, 1),
                "curvature": 0.5,
                "solidity": 0.9,
                "extent": 0.8,
            }
        
        # Detect spots
        if self.enable_texture:
            spot_count, spot_ratio = self.detect_spots(crop)
            texture_variance = self.compute_texture_variance(crop)
        else:
            spot_count, spot_ratio = 0, 0.0
            texture_variance = 0.1
        
        # Compute overall quality score
        quality_score = self._compute_quality_score(
            color_features, morph_features, spot_count, spot_ratio, texture_variance
        )
        
        return BananaFeatures(
            yellow_ratio=color_features["yellow_ratio"],
            green_ratio=color_features["green_ratio"],
            brown_ratio=color_features["brown_ratio"],
            color_uniformity=color_features["color_uniformity"],
            area=morph_features["area"],
            perimeter=morph_features["perimeter"],
            aspect_ratio=morph_features["aspect_ratio"],
            curvature=morph_features["curvature"],
            solidity=morph_features["solidity"],
            texture_variance=texture_variance,
            spot_count=spot_count,
            quality_score=quality_score,
        )
    
    def _compute_quality_score(
        self,
        color_features: Dict[str, float],
        morph_features: Dict[str, float],
        spot_count: int,
        spot_ratio: float,
        texture_variance: float,
    ) -> float:
        """Compute overall quality score from extracted features.
        
        Based on paper methodology: Weighted combination of color, morphology,
        and defect indicators.
        
        Quality Score Interpretation:
        - 0.8-1.0: Excellent (export quality)
        - 0.6-0.8: Good (ripe, ready to eat)
        - 0.4-0.6: Fair (slightly overripe)
        - 0.2-0.4: Poor (overripe with spots)
        - 0.0-0.2: Bad (defective/rotten)
        """
        # Color score (yellow + some green is ideal, brown/black is bad)
        yellow = color_features.get("yellow_ratio", 0.0)
        green = color_features.get("green_ratio", 0.0)
        brown = color_features.get("brown_ratio", 0.0)
        black = color_features.get("black_ratio", 0.0)
        uniformity = color_features.get("color_uniformity", 0.5)
        
        # Ideal is mostly yellow with good uniformity
        color_score = yellow * 0.5 + green * 0.3 + uniformity * 0.3 - brown * 0.5 - black * 1.0
        color_score = max(0.0, min(1.0, color_score))
        
        # Shape score (typical banana has aspect ratio 2-4, good solidity)
        aspect = morph_features.get("aspect_ratio", 1.0)
        solidity = morph_features.get("solidity", 0.8)
        
        # Penalize unusual aspect ratios
        if 0.2 <= aspect <= 0.6:
            aspect_score = 1.0  # Vertical banana (normal)
        elif 1.5 <= aspect <= 4.0:
            aspect_score = 1.0  # Horizontal banana (normal)
        else:
            aspect_score = 0.7
        
        shape_score = aspect_score * 0.4 + solidity * 0.6
        
        # Defect score (fewer spots = better)
        spot_penalty = min(1.0, spot_count * 0.05 + spot_ratio * 5.0)
        texture_penalty = texture_variance * 0.3  # High variance might indicate damage
        defect_score = max(0.0, 1.0 - spot_penalty - texture_penalty)
        
        # Weighted combination
        quality_score = (
            color_score * 0.45 +
            shape_score * 0.25 +
            defect_score * 0.30
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _default_features(self) -> BananaFeatures:
        """Return default features when analysis fails."""
        return BananaFeatures(
            yellow_ratio=0.0,
            green_ratio=0.0,
            brown_ratio=0.0,
            color_uniformity=0.0,
            area=0.0,
            perimeter=0.0,
            aspect_ratio=0.0,
            curvature=0.0,
            solidity=0.0,
            texture_variance=0.0,
            spot_count=0,
            quality_score=0.0,
        )
    
    def refine_category_with_features(
        self,
        predicted_category: str,
        features: BananaFeatures,
        cls_confidence: float,
    ) -> Tuple[str, float]:
        """Refine category prediction using extracted features.
        
        Based on paper: Ensemble approach combining DL prediction with
        traditional CV features improves accuracy.
        
        Args:
            predicted_category: Category from YOLO classifier
            features: Extracted banana features
            cls_confidence: Original classifier confidence
            
        Returns:
            Tuple of (refined_category, refined_confidence)
        """
        # Feature-based category estimation
        feature_category = self._estimate_category_from_features(features)
        feature_confidence = self._estimate_confidence_from_features(features)
        
        # If classifier is very confident (>0.85), trust it mostly
        if cls_confidence > 0.85:
            return predicted_category, cls_confidence
        
        # If classifier is uncertain (<0.5), rely more on features
        if cls_confidence < 0.5:
            # Weight features more heavily
            if feature_confidence > 0.6:
                return feature_category, (cls_confidence + feature_confidence) / 2
        
        # For medium confidence, use weighted ensemble
        # If both agree, boost confidence
        if feature_category == predicted_category:
            ensemble_conf = min(1.0, cls_confidence * 0.7 + feature_confidence * 0.3 + 0.1)
            return predicted_category, ensemble_conf
        
        # If they disagree, use the one with higher confidence
        if feature_confidence > cls_confidence:
            return feature_category, (cls_confidence + feature_confidence) / 2
        else:
            return predicted_category, cls_confidence
    
    def _estimate_category_from_features(self, features: BananaFeatures) -> str:
        """Estimate category purely from extracted features."""
        # High green ratio = unripe
        if features.green_ratio > 0.4:
            return "unripe"
        
        # High brown/spots = overripe or defective
        if features.brown_ratio > 0.3 or features.spot_count > 20:
            return "defective"
        
        if features.brown_ratio > 0.15 or features.spot_count > 10:
            return "overripe"
        
        # High yellow with good uniformity = export quality
        if features.yellow_ratio > 0.5 and features.color_uniformity > 0.6:
            return "export"
        
        # Moderate yellow = ripe (export)
        if features.yellow_ratio > 0.3:
            return "export"
        
        # Low color detection = uncertain
        return "export"  # Default to export
    
    def _estimate_confidence_from_features(self, features: BananaFeatures) -> float:
        """Estimate confidence based on feature clarity."""
        # More distinct features = higher confidence
        max_color = max(features.yellow_ratio, features.green_ratio, features.brown_ratio)
        
        # Clear dominant color = higher confidence
        if max_color > 0.5:
            confidence = 0.7 + max_color * 0.3
        elif max_color > 0.3:
            confidence = 0.5 + max_color * 0.3
        else:
            confidence = 0.3 + max_color * 0.3
        
        # Good shape contributes to confidence
        if 0.7 < features.solidity < 1.0:
            confidence += 0.1
        
        return min(1.0, confidence)


class MultiScaleDetector:
    """Multi-scale detection for improved banana detection accuracy.
    
    Based on paper: Multi-scale detection with different input sizes
    improves detection of bananas at various distances.
    """
    
    def __init__(
        self,
        scales: List[float] = None,
        nms_threshold: float = 0.4,
    ):
        """Initialize multi-scale detector.
        
        Args:
            scales: List of scale factors (default: [0.5, 1.0, 1.5])
            nms_threshold: NMS IoU threshold for merging detections
        """
        self.scales = scales or [0.5, 1.0, 1.5]
        self.nms_threshold = nms_threshold
    
    def scale_frame(
        self,
        frame_bgr: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Scale frame by given factor."""
        if cv2 is None or scale == 1.0:
            return frame_bgr
        
        h, w = frame_bgr.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if scale < 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=interpolation)
    
    def scale_bbox(
        self,
        bbox_xyxy: Tuple[int, int, int, int],
        scale: float,
    ) -> Tuple[int, int, int, int]:
        """Scale bbox coordinates back to original frame size."""
        x1, y1, x2, y2 = bbox_xyxy
        return (
            int(x1 / scale),
            int(y1 / scale),
            int(x2 / scale),
            int(y2 / scale),
        )
    
    @staticmethod
    def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def merge_detections(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Merge detections from multiple scales using NMS.
        
        Args:
            detections: List of (bbox_xyxy, confidence) tuples
            
        Returns:
            Filtered list of detections after NMS
        """
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Filter out overlapping detections
            remaining = []
            for det in detections:
                iou = self.compute_iou(best[0], det[0])
                if iou < self.nms_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
