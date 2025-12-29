"""Test script for the enhanced banana analyzer module.

Run this script to verify the BananaAnalyzer is working correctly.

Usage:
    python test_analyzer.py [image_path]

If no image path is provided, it will create a synthetic test image.
"""

from __future__ import annotations

import sys
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV not installed. Some tests will be skipped.")

from app.banana_analyzer import BananaAnalyzer, BananaFeatures, MultiScaleDetector


def create_synthetic_banana_image(
    width: int = 300,
    height: int = 200,
    ripeness: str = "ripe",
) -> np.ndarray:
    """Create a synthetic banana-like image for testing."""
    if not HAS_CV2:
        # Return a simple numpy array
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark gray background
    
    # Draw banana-shaped ellipse
    center = (width // 2, height // 2)
    axes = (width // 3, height // 4)
    
    # Color based on ripeness
    if ripeness == "unripe":
        color = (0, 180, 50)  # Green
    elif ripeness == "ripe":
        color = (0, 200, 255)  # Yellow (BGR)
    elif ripeness == "overripe":
        color = (20, 140, 200)  # Orange-brown
    else:  # defective
        color = (30, 50, 80)  # Dark brown
    
    cv2.ellipse(img, center, axes, 30, 0, 360, color, -1)
    
    # Add some spots for overripe/defective
    if ripeness in ("overripe", "defective"):
        num_spots = 15 if ripeness == "defective" else 5
        for _ in range(num_spots):
            spot_x = np.random.randint(center[0] - axes[0], center[0] + axes[0])
            spot_y = np.random.randint(center[1] - axes[1], center[1] + axes[1])
            spot_radius = np.random.randint(3, 8)
            cv2.circle(img, (spot_x, spot_y), spot_radius, (20, 30, 40), -1)
    
    return img


def test_analyzer_basic():
    """Test basic analyzer functionality."""
    print("\n" + "="*60)
    print("Testing BananaAnalyzer Basic Functionality")
    print("="*60)
    
    analyzer = BananaAnalyzer(
        enable_color_analysis=True,
        enable_morphology=True,
        enable_texture=True,
    )
    
    # Test with synthetic images
    test_cases = ["unripe", "ripe", "overripe", "defective"]
    
    for ripeness in test_cases:
        print(f"\n--- Testing {ripeness} banana ---")
        
        img = create_synthetic_banana_image(ripeness=ripeness)
        features = analyzer.analyze(img)
        
        print(f"  Quality Score: {features.quality_score:.2%}")
        print(f"  Yellow Ratio:  {features.yellow_ratio:.2%}")
        print(f"  Green Ratio:   {features.green_ratio:.2%}")
        print(f"  Brown Ratio:   {features.brown_ratio:.2%}")
        print(f"  Spot Count:    {features.spot_count}")
        print(f"  Solidity:      {features.solidity:.2%}")
        
        # Verify reasonable values
        assert 0 <= features.quality_score <= 1, "Quality score out of range"
        assert 0 <= features.yellow_ratio <= 1, "Yellow ratio out of range"
        assert features.spot_count >= 0, "Spot count should be non-negative"
        
        # Test category estimation
        estimated_cat = analyzer._estimate_category_from_features(features)
        print(f"  Estimated Category: {estimated_cat}")
    
    print("\n✓ Basic tests passed!")


def test_preprocessing():
    """Test preprocessing functions."""
    print("\n" + "="*60)
    print("Testing Preprocessing Functions")
    print("="*60)
    
    if not HAS_CV2:
        print("  [SKIP] OpenCV not available")
        return
    
    analyzer = BananaAnalyzer()
    img = create_synthetic_banana_image(ripeness="ripe")
    
    # Test contrast enhancement
    enhanced = analyzer.preprocess_frame(img, enhance_contrast=True, denoise=True)
    assert enhanced.shape == img.shape, "Shape should remain same after preprocessing"
    print("  ✓ Preprocessing works correctly")
    
    # Test banana mask creation
    mask = analyzer.create_banana_mask(img)
    assert mask.shape[:2] == img.shape[:2], "Mask should have same spatial dimensions"
    assert mask.dtype == np.uint8, "Mask should be uint8"
    print("  ✓ Banana mask creation works")


def test_multi_scale_detector():
    """Test multi-scale detection utilities."""
    print("\n" + "="*60)
    print("Testing MultiScaleDetector")
    print("="*60)
    
    detector = MultiScaleDetector(scales=[0.5, 1.0, 1.5])
    
    # Test IoU computation
    box1 = (0, 0, 100, 100)
    box2 = (50, 50, 150, 150)
    iou = detector.compute_iou(box1, box2)
    print(f"  IoU of overlapping boxes: {iou:.2%}")
    assert 0 < iou < 1, "IoU should be between 0 and 1 for overlapping boxes"
    
    # Test perfect overlap
    iou_same = detector.compute_iou(box1, box1)
    assert abs(iou_same - 1.0) < 0.001, "IoU of same box should be 1.0"
    print(f"  IoU of same box: {iou_same:.2%}")
    
    # Test no overlap
    box3 = (200, 200, 300, 300)
    iou_none = detector.compute_iou(box1, box3)
    assert iou_none == 0, "IoU of non-overlapping boxes should be 0"
    print(f"  IoU of non-overlapping boxes: {iou_none:.2%}")
    
    # Test NMS
    detections = [
        ((0, 0, 100, 100), 0.9),
        ((10, 10, 110, 110), 0.8),  # Overlaps with first
        ((200, 200, 300, 300), 0.7),  # No overlap
    ]
    merged = detector.merge_detections(detections)
    print(f"  NMS reduced {len(detections)} detections to {len(merged)}")
    assert len(merged) == 2, "NMS should keep 2 non-overlapping detections"
    
    print("  ✓ MultiScaleDetector tests passed!")


def test_feature_refinement():
    """Test category refinement using features."""
    print("\n" + "="*60)
    print("Testing Feature-based Category Refinement")
    print("="*60)
    
    analyzer = BananaAnalyzer()
    
    # Create a clearly green banana
    green_img = create_synthetic_banana_image(ripeness="unripe")
    features = analyzer.analyze(green_img)
    
    # Test refinement when classifier says "export" but features say "unripe"
    refined_cat, refined_conf = analyzer.refine_category_with_features(
        "export",  # Classifier prediction
        features,
        0.4,  # Low confidence
    )
    print(f"  Original: export (40% conf)")
    print(f"  Refined:  {refined_cat} ({refined_conf:.0%} conf)")
    
    # Create a clearly yellow banana
    yellow_img = create_synthetic_banana_image(ripeness="ripe")
    features_yellow = analyzer.analyze(yellow_img)
    
    refined_cat2, refined_conf2 = analyzer.refine_category_with_features(
        "export",
        features_yellow,
        0.9,  # High confidence
    )
    print(f"\n  Original: export (90% conf)")
    print(f"  Refined:  {refined_cat2} ({refined_conf2:.0%} conf)")
    
    print("  ✓ Feature refinement tests passed!")


def test_with_real_image(image_path: str):
    """Test analyzer with a real image."""
    print("\n" + "="*60)
    print(f"Testing with real image: {image_path}")
    print("="*60)
    
    if not HAS_CV2:
        print("  [SKIP] OpenCV not available")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Could not load image: {image_path}")
        return
    
    analyzer = BananaAnalyzer()
    
    # Preprocess
    processed = analyzer.preprocess_frame(img)
    
    # Analyze
    features = analyzer.analyze(processed)
    
    print(f"\n  Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"  Quality Score: {features.quality_score:.2%}")
    print(f"  Yellow Ratio:  {features.yellow_ratio:.2%}")
    print(f"  Green Ratio:   {features.green_ratio:.2%}")
    print(f"  Brown Ratio:   {features.brown_ratio:.2%}")
    print(f"  Color Uniformity: {features.color_uniformity:.2%}")
    print(f"  Spot Count:    {features.spot_count}")
    print(f"  Solidity:      {features.solidity:.2%}")
    print(f"  Aspect Ratio:  {features.aspect_ratio:.2f}")
    print(f"  Texture Variance: {features.texture_variance:.2%}")
    
    estimated_cat = analyzer._estimate_category_from_features(features)
    print(f"\n  Estimated Category: {estimated_cat}")


def main():
    print("="*60)
    print("Banana Analyzer Test Suite")
    print("Based on: 'Comparative Analysis of Banana Detection Models'")
    print("="*60)
    
    # Run basic tests
    test_analyzer_basic()
    test_preprocessing()
    test_multi_scale_detector()
    test_feature_refinement()
    
    # Test with real image if provided
    if len(sys.argv) > 1:
        test_with_real_image(sys.argv[1])
    
    print("\n" + "="*60)
    print("All tests completed successfully! ✓")
    print("="*60)


if __name__ == "__main__":
    main()
