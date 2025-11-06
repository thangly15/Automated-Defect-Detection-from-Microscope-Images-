"""
Visualization Module

Creates annotated images with highlighted defects and generates visual reports.
"""

import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path
import colorsys

from src.defect_detector import Defect


class DefectVisualizer:
    """
    Visualizes detected defects on images with annotations.

    Provides multiple visualization modes:
    - Bounding boxes
    - Contour overlays
    - Heatmaps
    - Side-by-side comparisons
    """

    # Color scheme for different defect types
    DEFECT_COLORS = {
        'scratch': (0, 0, 255),          # Red
        'void': (255, 0, 0),              # Blue
        'particle': (0, 255, 0),          # Green
        'edge_defect': (0, 165, 255),     # Orange
        'texture_anomaly': (255, 0, 255), # Magenta
        'threshold_anomaly': (255, 255, 0) # Cyan
    }

    def __init__(self, visualization_mode: str = 'bbox'):
        """
        Initialize the visualizer.

        Args:
            visualization_mode: 'bbox', 'contour', 'both', or 'heatmap'
        """
        self.visualization_mode = visualization_mode

    def annotate_image(self, image: np.ndarray, defects: List[Defect],
                       show_labels: bool = True) -> np.ndarray:
        """
        Create an annotated image with defects highlighted.

        Args:
            image: Original image (BGR or grayscale)
            defects: List of detected defects
            show_labels: Whether to show defect type labels

        Returns:
            Annotated image
        """
        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        # Sort defects by area (largest first) for better visualization
        defects = sorted(defects, key=lambda d: d.area, reverse=True)

        if self.visualization_mode == 'bbox':
            annotated = self._draw_bboxes(annotated, defects, show_labels)
        elif self.visualization_mode == 'contour':
            annotated = self._draw_contours(annotated, defects)
        elif self.visualization_mode == 'both':
            annotated = self._draw_contours(annotated, defects)
            annotated = self._draw_bboxes(annotated, defects, show_labels)
        elif self.visualization_mode == 'heatmap':
            annotated = self._create_heatmap(annotated, defects)

        # Add summary text
        annotated = self._add_summary(annotated, defects)

        return annotated

    def _draw_bboxes(self, image: np.ndarray, defects: List[Defect],
                     show_labels: bool = True) -> np.ndarray:
        """Draw bounding boxes around defects"""
        annotated = image.copy()

        for i, defect in enumerate(defects, 1):
            x, y, w, h = defect.bbox
            color = self.DEFECT_COLORS.get(defect.defect_type, (255, 255, 255))

            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Add label
            if show_labels:
                label = f"{i}: {defect.defect_type}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               0.5, 1)

                # Draw label background
                cv2.rectangle(annotated,
                            (x, y - label_size[1] - 5),
                            (x + label_size[0], y),
                            color, -1)

                # Draw label text
                cv2.putText(annotated, label, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def _draw_contours(self, image: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """Draw contours of defects"""
        annotated = image.copy()

        for defect in defects:
            color = self.DEFECT_COLORS.get(defect.defect_type, (255, 255, 255))
            cv2.drawContours(annotated, [defect.contour], -1, color, 2)

        return annotated

    def _create_heatmap(self, image: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """Create a heatmap overlay showing defect density"""
        # Create defect mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for defect in defects:
            cv2.drawContours(mask, [defect.contour], -1, 255, -1)

        # Apply Gaussian blur to create heatmap effect
        heatmap = cv2.GaussianBlur(mask, (51, 51), 0)

        # Convert to color heatmap (red = high defect density)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend with original image
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        annotated = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)

        return annotated

    def _add_summary(self, image: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """Add summary information to the image"""
        annotated = image.copy()

        # Count defects by type
        defect_counts = {}
        for defect in defects:
            defect_counts[defect.defect_type] = defect_counts.get(defect.defect_type, 0) + 1

        # Create summary text
        summary = f"Total Defects: {len(defects)}"
        y_offset = 30

        # Draw semi-transparent background for summary
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, 30 + 25 * (len(defect_counts) + 1)),
                     (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)

        # Draw summary text
        cv2.putText(annotated, summary, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset += 25
        for defect_type, count in sorted(defect_counts.items()):
            color = self.DEFECT_COLORS.get(defect_type, (255, 255, 255))
            text = f"  {defect_type}: {count}"
            cv2.putText(annotated, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 25

        return annotated

    def create_comparison(self, original: np.ndarray, annotated: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of original and annotated images.

        Args:
            original: Original image
            annotated: Annotated image

        Returns:
            Combined side-by-side image
        """
        # Ensure both images are BGR
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        # Resize to same dimensions if needed
        if original.shape != annotated.shape:
            annotated = cv2.resize(annotated, (original.shape[1], original.shape[0]))

        # Add labels
        cv2.putText(original, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, "Defects Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Concatenate horizontally
        comparison = np.hstack([original, annotated])

        return comparison

    def save_annotated_image(self, image: np.ndarray, defects: List[Defect],
                            output_path: str, show_labels: bool = True) -> None:
        """
        Save annotated image to file.

        Args:
            image: Original image
            defects: List of detected defects
            output_path: Path to save the annotated image
            show_labels: Whether to show defect type labels
        """
        annotated = self.annotate_image(image, defects, show_labels)
        cv2.imwrite(output_path, annotated)

    def create_report_image(self, original: np.ndarray, defects: List[Defect],
                           metadata: dict) -> np.ndarray:
        """
        Create a comprehensive report image with multiple views.

        Args:
            original: Original image
            defects: List of detected defects
            metadata: Image metadata

        Returns:
            Report image with multiple visualizations
        """
        # Create different visualizations
        viz_bbox = DefectVisualizer('bbox')
        viz_heatmap = DefectVisualizer('heatmap')

        annotated_bbox = viz_bbox.annotate_image(original, defects)
        annotated_heatmap = viz_heatmap.annotate_image(original, defects)

        # Ensure all images are BGR
        if len(original.shape) == 2:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            original_bgr = original.copy()

        # Add titles
        cv2.putText(original_bgr, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_bbox, "Detected Defects", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_heatmap, "Defect Heatmap", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Create grid layout
        top_row = np.hstack([original_bgr, annotated_bbox])

        # Create info panel
        info_panel = self._create_info_panel(original.shape, defects, metadata)
        bottom_row = np.hstack([annotated_heatmap, info_panel])

        # Combine
        report = np.vstack([top_row, bottom_row])

        return report

    def _create_info_panel(self, image_shape: Tuple, defects: List[Defect],
                          metadata: dict) -> np.ndarray:
        """Create an information panel with statistics"""
        # Create blank panel
        height, width = image_shape[:2]
        panel = np.ones((height, width, 3), dtype=np.uint8) * 240

        # Add title
        y_offset = 40
        cv2.putText(panel, "Detection Statistics", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        y_offset += 40

        # Add statistics
        stats = [
            f"Image: {metadata.get('filename', 'N/A')}",
            f"Dimensions: {image_shape[1]}x{image_shape[0]}",
            f"",
            f"Total Defects: {len(defects)}",
            f"",
            "Defects by Type:",
        ]

        # Count defects by type
        defect_counts = {}
        for defect in defects:
            defect_counts[defect.defect_type] = defect_counts.get(defect.defect_type, 0) + 1

        for defect_type, count in sorted(defect_counts.items()):
            stats.append(f"  {defect_type}: {count}")

        stats.extend([
            "",
            "Image Quality Metrics:",
            f"Focus: {metadata.get('focus_quality', 0):.2f}",
            f"Contrast: {metadata.get('contrast', 0):.2f}",
            f"Noise Level: {metadata.get('noise_level', 0):.2f}",
        ])

        # Draw statistics
        for line in stats:
            cv2.putText(panel, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25

        return panel
