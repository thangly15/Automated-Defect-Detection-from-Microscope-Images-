#!/usr/bin/env python3
"""
SIMPLE DEFECT DETECTION SCRIPT - BEGINNER FRIENDLY!

This is the easiest way to detect defects in your microscope images.
Just run this script and follow the prompts!

HOW TO RUN:
1. Put your images in a folder
2. Run: python simple_detect.py
3. Enter your folder path when asked
4. Check the results folder!

OR use command line:
    python simple_detect.py my_images_folder/
"""

import cv2
import os
import sys

# Import our detector
from src.defect_detector import create_detector


def main():
    """
    Main function - keeps things simple!
    """
    print("=" * 70)
    print("MICROSCOPE IMAGE DEFECT DETECTOR")
    print("Simple Version for Beginners")
    print("=" * 70)
    print()

    # =================================================================
    # STEP 1: Get the folder with images
    # =================================================================

    if len(sys.argv) > 1:
        # User provided folder as command line argument
        input_folder = sys.argv[1]
    else:
        # Ask user for folder
        print("Enter the path to your images folder:")
        print("(Example: my_images/ or C:/Users/YourName/Desktop/images/)")
        input_folder = input("Folder path: ").strip()

    # Check if folder exists
    if not os.path.exists(input_folder):
        print(f"\nERROR: Folder '{input_folder}' not found!")
        print("Please check the path and try again.")
        return

    print(f"\nLooking for images in: {input_folder}")

    # =================================================================
    # STEP 2: Find all image files
    # =================================================================

    image_files = []
    supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

    for filename in os.listdir(input_folder):
        # Check if file has image extension
        file_lower = filename.lower()
        if any(file_lower.endswith(fmt) for fmt in supported_formats):
            full_path = os.path.join(input_folder, filename)
            image_files.append(full_path)

    if not image_files:
        print("\nNo image files found!")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return

    print(f"Found {len(image_files)} images")
    print()

    # =================================================================
    # STEP 3: Ask user about settings
    # =================================================================

    print("Do you want to:")
    print("  1. Use default settings (recommended for beginners)")
    print("  2. Custom settings (adjust sensitivity)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        print("\nCustom Settings:")
        print("Minimum defect size (pixels) - smaller = more sensitive")
        min_area_input = input("  Min size [default: 10]: ").strip()
        min_area = int(min_area_input) if min_area_input else 10

        print("Use ensemble mode? (more accurate but slower)")
        ensemble_input = input("  Ensemble (y/n) [default: y]: ").strip().lower()
        use_ensemble = ensemble_input != 'n'
    else:
        # Default settings
        min_area = 10
        use_ensemble = True

    # =================================================================
    # STEP 4: Create output folder
    # =================================================================

    output_folder = "results_simple"
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nResults will be saved to: {output_folder}/")
    print()

    # =================================================================
    # STEP 5: Create detector and process images
    # =================================================================

    print("Creating detector...")
    detector = create_detector(min_area=min_area, use_ensemble=use_ensemble)
    print()

    # Keep track of results
    total_defects = 0
    results_summary = []

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"[{i}/{len(image_files)}] Processing: {filename}")

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ERROR: Could not load {filename}")
                continue

            # Detect defects
            defects = detector.detect_defects(image)
            total_defects += len(defects)

            # Save results
            results_summary.append({
                'filename': filename,
                'defect_count': len(defects),
                'defects': defects
            })

            # Draw defects on image
            annotated = draw_defects_simple(image, defects)

            # Save annotated image
            output_path = os.path.join(output_folder, f"result_{filename}")
            cv2.imwrite(output_path, annotated)
            print(f"  ✓ Found {len(defects)} defects")
            print(f"  ✓ Saved to: {output_path}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")

        print()

    # =================================================================
    # STEP 6: Save summary report
    # =================================================================

    report_path = os.path.join(output_folder, "summary.txt")
    with open(report_path, 'w') as f:
        f.write("DEFECT DETECTION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Images: {len(image_files)}\n")
        f.write(f"Total Defects Found: {total_defects}\n\n")
        f.write("Per-Image Results:\n")
        f.write("-" * 70 + "\n")

        for result in results_summary:
            f.write(f"\n{result['filename']}\n")
            f.write(f"  Defects: {result['defect_count']}\n")

            # Count by type
            defect_types = {}
            for defect in result['defects']:
                defect_types[defect.defect_type] = defect_types.get(defect.defect_type, 0) + 1

            if defect_types:
                f.write("  Types:\n")
                for dtype, count in defect_types.items():
                    f.write(f"    {dtype}: {count}\n")

    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nProcessed {len(image_files)} images")
    print(f"Found {total_defects} total defects")
    print(f"\nResults saved to: {output_folder}/")
    print(f"Summary report: {report_path}")
    print()


def draw_defects_simple(image, defects):
    """
    Draw boxes around defects on the image.

    Makes a copy so we don't modify the original.
    """
    # Make a copy
    annotated = image.copy()

    # Colors for different defect types
    colors = {
        'scratch': (0, 0, 255),          # Red
        'void': (255, 0, 0),              # Blue
        'particle': (0, 255, 0),          # Green
        'edge_defect': (0, 165, 255),     # Orange
        'texture_anomaly': (255, 0, 255), # Magenta
        'threshold_anomaly': (255, 255, 0) # Cyan
    }

    # Draw each defect
    for defect in defects:
        x, y, w, h = defect.bbox

        # Get color for this defect type
        color = colors.get(defect.defect_type, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    # Add count at top
    text = f"Defects Found: {len(defects)}"
    cv2.putText(annotated, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated


# =================================================================
# RUN THE SCRIPT
# =================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install opencv-python numpy")
