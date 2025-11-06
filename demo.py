#!/usr/bin/env python3
"""
Demo Script for Defect Detection System

This script demonstrates basic usage of the defect detection system
with synthetic test images.
"""

import cv2
import numpy as np
from pathlib import Path

from src.batch_processor import BatchProcessor
from src.defect_detector import EnsembleDetector
from src.image_preprocessor import ImagePreprocessor
from src.visualizer import DefectVisualizer


def create_synthetic_test_image(width=800, height=600):
    """
    Create a synthetic microscope image with artificial defects.

    This is useful for testing and demonstration when real microscope
    images are not available.
    """
    # Create base image (simulate microscope background)
    image = np.ones((height, width), dtype=np.uint8) * 200

    # Add some texture
    noise = np.random.normal(0, 10, (height, width))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Add synthetic defects

    # 1. Scratch (dark line)
    cv2.line(image, (100, 150), (700, 200), 80, 3)

    # 2. Void (dark spot)
    cv2.circle(image, (400, 300), 25, 50, -1)

    # 3. Particle (bright spot)
    cv2.circle(image, (250, 450), 15, 255, -1)

    # 4. Edge defect (irregular shape)
    pts = np.array([[600, 400], [650, 380], [680, 420], [640, 460]], np.int32)
    cv2.fillPoly(image, [pts], 100)

    # 5. Another scratch
    cv2.line(image, (150, 500), (400, 520), 70, 2)

    # 6. Small particle cluster
    cv2.circle(image, (550, 150), 8, 245, -1)
    cv2.circle(image, (565, 155), 6, 250, -1)
    cv2.circle(image, (555, 165), 7, 240, -1)

    # 7. Void cluster
    cv2.circle(image, (200, 250), 12, 60, -1)
    cv2.circle(image, (220, 255), 10, 55, -1)

    # Add slight blur to simulate microscope optics
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def run_demo():
    """Run demonstration of the defect detection system"""
    print("=" * 70)
    print("Defect Detection System - Demo")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    print("1. Creating synthetic test image...")
    test_image = create_synthetic_test_image()

    # Save test image
    input_path = output_dir / "test_image.png"
    cv2.imwrite(str(input_path), test_image)
    print(f"   Saved test image to: {input_path}")
    print()

    print("2. Initializing defect detection system...")
    # Create detector components
    detector = EnsembleDetector()
    preprocessor = ImagePreprocessor()
    visualizer = DefectVisualizer(visualization_mode='both')

    # Create processor
    processor = BatchProcessor(detector, preprocessor, visualizer)
    print("   System initialized successfully")
    print()

    print("3. Processing image...")
    result = processor.process_single_image(
        str(input_path),
        str(output_dir),
        save_annotated=True
    )
    print(f"   Processing complete!")
    print()

    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Image: {result['image_path']}")
    print(f"Defects Detected: {result['defect_count']}")
    print()

    if result['defect_count'] > 0:
        # Count by type
        defect_counts = {}
        for defect in result['defects']:
            defect_counts[defect.defect_type] = defect_counts.get(defect.defect_type, 0) + 1

        print("Breakdown by Type:")
        for defect_type, count in sorted(defect_counts.items()):
            print(f"  {defect_type:.<40} {count:>3}")
        print()

        print("Detected Defects:")
        for i, defect in enumerate(result['defects'], 1):
            x, y, w, h = defect.bbox
            print(f"  {i}. {defect.defect_type} at ({x}, {y}), "
                  f"size: {w}x{h}, area: {defect.area:.1f}px²")

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print()
    print(f"Original Image: {input_path}")
    print(f"Annotated Image: {result['annotated_path']}")
    print()
    print(f"All outputs saved to: {output_dir}/")
    print()

    # Image quality metrics
    print("=" * 70)
    print("IMAGE QUALITY METRICS")
    print("=" * 70)
    print()
    metadata = result['metadata']
    print(f"Focus Quality: {metadata.get('focus_quality', 0):.2%}")
    print(f"Contrast: {metadata.get('contrast', 0):.2%}")
    print(f"Noise Level: {metadata.get('noise_level', 0):.2%}")
    print(f"Mean Intensity: {metadata.get('mean_intensity', 0):.1f}")
    print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. View the annotated image in demo_output/")
    print("  2. Try with your own microscope images:")
    print("     python detect_defects.py -i your_image.jpg -o results/")
    print("  3. Launch the web interface:")
    print("     streamlit run streamlit_app.py")
    print()


if __name__ == '__main__':
    try:
        run_demo()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
