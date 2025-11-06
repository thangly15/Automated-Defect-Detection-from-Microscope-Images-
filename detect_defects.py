#!/usr/bin/env python3
"""
Automated Defect Detection System - Main CLI Interface

This script provides a command-line interface for detecting defects in microscope images.

Usage:
    python detect_defects.py --input <input_path> --output <output_path> [options]

Examples:
    # Process a single image
    python detect_defects.py --input image.jpg --output results/

    # Process all images in a directory
    python detect_defects.py --input images/ --output results/

    # Process with custom configuration
    python detect_defects.py --input images/ --output results/ --config config.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging

from src.batch_processor import BatchProcessor, create_batch_processor
from src.defect_detector import DefectDetector, EnsembleDetector
from src.image_preprocessor import ImagePreprocessor
from src.visualizer import DefectVisualizer
from src.report_generator import ReportGenerator, Statistics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Automated Defect Detection for Microscope Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single image:
    python detect_defects.py --input image.jpg --output results/

  Process directory of images:
    python detect_defects.py --input images/ --output results/

  Process with custom settings:
    python detect_defects.py --input images/ --output results/ --ensemble --min-area 20

For more information, see the documentation.
        """
    )

    # Required arguments
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input image file or directory'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for results'
    )

    # Optional arguments
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--ensemble',
        action='store_true',
        default=True,
        help='Use ensemble detection for higher accuracy (default: True)'
    )

    parser.add_argument(
        '--no-ensemble',
        action='store_true',
        help='Disable ensemble detection (faster but may miss some defects)'
    )

    parser.add_argument(
        '--min-area',
        type=float,
        default=10,
        help='Minimum defect area in pixels (default: 10)'
    )

    parser.add_argument(
        '--max-area',
        type=float,
        default=50000,
        help='Maximum defect area in pixels (default: 50000)'
    )

    parser.add_argument(
        '--visualization',
        choices=['bbox', 'contour', 'both', 'heatmap'],
        default='both',
        help='Visualization mode (default: both)'
    )

    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip generating CSV/JSON reports'
    )

    parser.add_argument(
        '--no-annotated',
        action='store_true',
        help='Skip saving annotated images'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Defect Detection System v1.0.0'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def create_config_from_args(args) -> dict:
    """Create configuration dictionary from command-line arguments"""
    config = {
        'use_ensemble': args.ensemble and not args.no_ensemble,
        'visualization_mode': args.visualization,
        'detector': {
            'min_defect_area': args.min_area,
            'max_defect_area': args.max_area
        }
    }

    return config


def print_summary(summary: dict):
    """Print processing summary to console"""
    print("\n" + "=" * 70)
    print("DEFECT DETECTION SUMMARY")
    print("=" * 70)

    print(f"\nImages Processed: {summary['processed_images']}/{summary['total_images']}")

    if summary['failed_images'] > 0:
        print(f"Failed Images: {summary['failed_images']}")
        for filename, error in summary['failed']:
            print(f"  - {filename}: {error}")

    print(f"\nTotal Defects Detected: {summary['total_defects']}")

    if summary['results']:
        # Calculate statistics
        stats = Statistics.calculate_statistics(summary['results'])

        print(f"Average Defects per Image: {stats['average_defects_per_image']:.1f}")

        print("\nDefects by Type:")
        for defect_type, count in sorted(stats['defect_types_distribution'].items(),
                                         key=lambda x: x[1], reverse=True):
            percentage = (count / summary['total_defects'] * 100) if summary['total_defects'] > 0 else 0
            print(f"  {defect_type:.<40} {count:>5} ({percentage:>5.1f}%)")

    if summary['report_paths']:
        print("\nGenerated Reports:")
        for report_type, path in summary['report_paths'].items():
            print(f"  {report_type}: {path}")

    print("\n" + "=" * 70)


def main():
    """Main entry point"""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Automated Defect Detection System")

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_config_from_args(args)

    # Check input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Create batch processor
    logger.info("Initializing defect detection system...")
    processor = create_batch_processor(config)

    # Process input
    try:
        if input_path.is_file():
            # Single image processing
            logger.info("Processing single image...")
            result = processor.process_single_image(
                str(input_path),
                args.output,
                save_annotated=not args.no_annotated
            )

            print("\n" + "=" * 70)
            print(f"Image: {Path(result['image_path']).name}")
            print(f"Defects Detected: {result['defect_count']}")

            if result['defect_count'] > 0:
                defect_counts = {}
                for defect in result['defects']:
                    defect_counts[defect.defect_type] = defect_counts.get(defect.defect_type, 0) + 1

                print("\nDefects by Type:")
                for defect_type, count in sorted(defect_counts.items()):
                    print(f"  {defect_type}: {count}")

            if result['annotated_path']:
                print(f"\nAnnotated image saved to: {result['annotated_path']}")

            print("=" * 70)

        elif input_path.is_dir():
            # Batch processing
            logger.info("Processing directory of images...")
            summary = processor.process_directory(
                str(input_path),
                args.output,
                generate_reports=not args.no_reports,
                save_annotated=not args.no_annotated
            )

            print_summary(summary)

        else:
            logger.error(f"Invalid input path: {args.input}")
            sys.exit(1)

        logger.info("Processing complete!")

    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
