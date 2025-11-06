"""
Report Generation Module

Generates structured reports (CSV, JSON) for defect detection results.
"""

import csv
import json
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.defect_detector import Defect


class ReportGenerator:
    """
    Generates structured reports from defect detection results.

    Supports multiple output formats:
    - CSV: Tabular data for spreadsheet analysis
    - JSON: Structured data for programmatic access
    - Summary report: Human-readable text summary
    """

    def __init__(self, output_dir: str):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_csv_report(self, results: Dict[str, List[Defect]],
                           filename: str = "defect_report.csv") -> str:
        """
        Generate CSV report of all detected defects.

        Args:
            results: Dictionary mapping image filenames to list of defects
            filename: Output CSV filename

        Returns:
            Path to the generated CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['image_name', 'defect_id', 'defect_type', 'bbox_x',
                         'bbox_y', 'bbox_width', 'bbox_height', 'area',
                         'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for image_name, defects in results.items():
                for i, defect in enumerate(defects, 1):
                    x, y, w, h = defect.bbox
                    writer.writerow({
                        'image_name': image_name,
                        'defect_id': i,
                        'defect_type': defect.defect_type,
                        'bbox_x': x,
                        'bbox_y': y,
                        'bbox_width': w,
                        'bbox_height': h,
                        'area': defect.area,
                        'confidence': defect.confidence
                    })

        return str(output_path)

    def generate_summary_csv(self, results: Dict[str, List[Defect]],
                            filename: str = "defect_summary.csv") -> str:
        """
        Generate summary CSV with defect counts per image.

        Args:
            results: Dictionary mapping image filenames to list of defects
            filename: Output CSV filename

        Returns:
            Path to the generated CSV file
        """
        output_path = self.output_dir / filename

        summary_data = []
        for image_name, defects in results.items():
            # Count defects by type
            defect_counts = {}
            for defect in defects:
                defect_counts[defect.defect_type] = defect_counts.get(defect.defect_type, 0) + 1

            row = {
                'image_name': image_name,
                'total_defects': len(defects),
                'scratches': defect_counts.get('scratch', 0),
                'voids': defect_counts.get('void', 0),
                'particles': defect_counts.get('particle', 0),
                'edge_defects': defect_counts.get('edge_defect', 0),
                'texture_anomalies': defect_counts.get('texture_anomaly', 0),
                'threshold_anomalies': defect_counts.get('threshold_anomaly', 0),
            }
            summary_data.append(row)

        # Write using pandas for cleaner output
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)

        return str(output_path)

    def generate_json_report(self, results: Dict[str, List[Defect]],
                            metadata: Dict[str, dict] = None,
                            filename: str = "defect_report.json") -> str:
        """
        Generate JSON report with detailed defect information.

        Args:
            results: Dictionary mapping image filenames to list of defects
            metadata: Optional dictionary mapping image filenames to metadata
            filename: Output JSON filename

        Returns:
            Path to the generated JSON file
        """
        output_path = self.output_dir / filename

        report = {
            'generated_at': datetime.now().isoformat(),
            'total_images': len(results),
            'total_defects': sum(len(defects) for defects in results.values()),
            'images': []
        }

        for image_name, defects in results.items():
            image_data = {
                'filename': image_name,
                'defect_count': len(defects),
                'defects': []
            }

            # Add metadata if available
            if metadata and image_name in metadata:
                image_data['metadata'] = {
                    'shape': metadata[image_name].get('shape'),
                    'focus_quality': metadata[image_name].get('focus_quality'),
                    'noise_level': metadata[image_name].get('noise_level'),
                    'contrast': metadata[image_name].get('contrast')
                }

            # Add defect details
            for i, defect in enumerate(defects, 1):
                x, y, w, h = defect.bbox
                defect_data = {
                    'id': i,
                    'type': defect.defect_type,
                    'bounding_box': {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    },
                    'area': float(defect.area),
                    'confidence': float(defect.confidence)
                }
                image_data['defects'].append(defect_data)

            report['images'].append(image_data)

        # Write JSON with pretty printing
        with open(output_path, 'w') as jsonfile:
            json.dump(report, jsonfile, indent=2)

        return str(output_path)

    def generate_text_summary(self, results: Dict[str, List[Defect]],
                             filename: str = "summary.txt") -> str:
        """
        Generate human-readable text summary.

        Args:
            results: Dictionary mapping image filenames to list of defects
            filename: Output text filename

        Returns:
            Path to the generated text file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as txtfile:
            txtfile.write("=" * 70 + "\n")
            txtfile.write("DEFECT DETECTION SUMMARY REPORT\n")
            txtfile.write("=" * 70 + "\n\n")

            txtfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            txtfile.write(f"Total Images Processed: {len(results)}\n")

            total_defects = sum(len(defects) for defects in results.values())
            txtfile.write(f"Total Defects Detected: {total_defects}\n\n")

            # Overall statistics
            txtfile.write("-" * 70 + "\n")
            txtfile.write("OVERALL STATISTICS\n")
            txtfile.write("-" * 70 + "\n\n")

            all_defect_types = {}
            for defects in results.values():
                for defect in defects:
                    all_defect_types[defect.defect_type] = all_defect_types.get(defect.defect_type, 0) + 1

            txtfile.write("Defects by Type:\n")
            for defect_type, count in sorted(all_defect_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_defects * 100) if total_defects > 0 else 0
                txtfile.write(f"  {defect_type:.<40} {count:>5} ({percentage:>5.1f}%)\n")

            txtfile.write("\n")

            # Per-image results
            txtfile.write("-" * 70 + "\n")
            txtfile.write("PER-IMAGE RESULTS\n")
            txtfile.write("-" * 70 + "\n\n")

            for image_name, defects in sorted(results.items()):
                txtfile.write(f"Image: {image_name}\n")
                txtfile.write(f"  Total Defects: {len(defects)}\n")

                if defects:
                    defect_counts = {}
                    for defect in defects:
                        defect_counts[defect.defect_type] = defect_counts.get(defect.defect_type, 0) + 1

                    txtfile.write("  Breakdown:\n")
                    for defect_type, count in sorted(defect_counts.items()):
                        txtfile.write(f"    {defect_type}: {count}\n")

                txtfile.write("\n")

            txtfile.write("=" * 70 + "\n")
            txtfile.write("END OF REPORT\n")
            txtfile.write("=" * 70 + "\n")

        return str(output_path)

    def generate_all_reports(self, results: Dict[str, List[Defect]],
                            metadata: Dict[str, dict] = None) -> Dict[str, str]:
        """
        Generate all report formats.

        Args:
            results: Dictionary mapping image filenames to list of defects
            metadata: Optional dictionary mapping image filenames to metadata

        Returns:
            Dictionary mapping report type to file path
        """
        report_paths = {
            'csv_detailed': self.generate_csv_report(results),
            'csv_summary': self.generate_summary_csv(results),
            'json': self.generate_json_report(results, metadata),
            'text': self.generate_text_summary(results)
        }

        return report_paths


class Statistics:
    """Utility class for calculating detection statistics"""

    @staticmethod
    def calculate_statistics(results: Dict[str, List[Defect]]) -> Dict:
        """
        Calculate comprehensive statistics from detection results.

        Args:
            results: Dictionary mapping image filenames to list of defects

        Returns:
            Dictionary with various statistics
        """
        total_images = len(results)
        total_defects = sum(len(defects) for defects in results.values())

        # Defects per image
        defects_per_image = [len(defects) for defects in results.values()]
        avg_defects = sum(defects_per_image) / total_images if total_images > 0 else 0

        # Defect types distribution
        defect_types = {}
        defect_areas = []
        for defects in results.values():
            for defect in defects:
                defect_types[defect.defect_type] = defect_types.get(defect.defect_type, 0) + 1
                defect_areas.append(defect.area)

        # Area statistics
        avg_area = sum(defect_areas) / len(defect_areas) if defect_areas else 0
        min_area = min(defect_areas) if defect_areas else 0
        max_area = max(defect_areas) if defect_areas else 0

        statistics = {
            'total_images': total_images,
            'total_defects': total_defects,
            'average_defects_per_image': avg_defects,
            'defect_types_distribution': defect_types,
            'defect_area': {
                'average': avg_area,
                'min': min_area,
                'max': max_area
            },
            'images_with_defects': sum(1 for defects in results.values() if len(defects) > 0),
            'images_without_defects': sum(1 for defects in results.values() if len(defects) == 0)
        }

        return statistics
