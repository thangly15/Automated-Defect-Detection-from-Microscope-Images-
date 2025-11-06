"""
Batch Processing Module

Handles batch processing of multiple microscope images.
"""

import cv2
from pathlib import Path
from typing import List, Dict, Optional, Callable
from tqdm import tqdm
import logging

from src.defect_detector import DefectDetector, Defect, EnsembleDetector
from src.image_preprocessor import ImagePreprocessor
from src.visualizer import DefectVisualizer
from src.report_generator import ReportGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes multiple images in batch for defect detection.

    Features:
    - Parallel processing support
    - Progress tracking
    - Automatic report generation
    - Error handling and recovery
    """

    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

    def __init__(self,
                 detector: DefectDetector = None,
                 preprocessor: ImagePreprocessor = None,
                 visualizer: DefectVisualizer = None,
                 use_ensemble: bool = True):
        """
        Initialize the batch processor.

        Args:
            detector: Defect detector instance (default: EnsembleDetector)
            preprocessor: Image preprocessor instance
            visualizer: Defect visualizer instance
            use_ensemble: Whether to use ensemble detection for better accuracy
        """
        self.detector = detector or (EnsembleDetector() if use_ensemble else DefectDetector())
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.visualizer = visualizer or DefectVisualizer(visualization_mode='both')

    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         generate_reports: bool = True,
                         save_annotated: bool = True,
                         progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output (annotated images and reports)
            generate_reports: Whether to generate CSV/JSON reports
            save_annotated: Whether to save annotated images
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with processing results and statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        if save_annotated:
            annotated_dir = output_path / 'annotated'
            annotated_dir.mkdir(exist_ok=True)

        # Find all image files
        image_files = self._find_image_files(input_path)

        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return {
                'status': 'no_images',
                'total_images': 0,
                'results': {},
                'metadata': {}
            }

        logger.info(f"Found {len(image_files)} images to process")

        # Process each image
        results = {}
        metadata = {}
        failed_images = []

        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Load and preprocess
                original, preprocessed, img_metadata = self.preprocessor.load_and_preprocess(
                    str(image_path)
                )

                # Detect defects
                defects = self.detector.detect_defects(preprocessed)

                # Store results
                image_name = image_path.name
                results[image_name] = defects
                metadata[image_name] = img_metadata

                logger.info(f"Processed {image_name}: {len(defects)} defects detected")

                # Save annotated image
                if save_annotated:
                    output_image_path = annotated_dir / f"annotated_{image_name}"
                    self.visualizer.save_annotated_image(
                        original, defects, str(output_image_path)
                    )

                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(image_files), image_name)

            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {str(e)}")
                failed_images.append((image_path.name, str(e)))

        # Generate reports
        report_paths = {}
        if generate_reports:
            logger.info("Generating reports...")
            report_gen = ReportGenerator(output_path)
            report_paths = report_gen.generate_all_reports(results, metadata)

        # Prepare summary
        summary = {
            'status': 'success',
            'total_images': len(image_files),
            'processed_images': len(results),
            'failed_images': len(failed_images),
            'total_defects': sum(len(defects) for defects in results.values()),
            'results': results,
            'metadata': metadata,
            'report_paths': report_paths,
            'failed': failed_images
        }

        logger.info(f"Processing complete: {len(results)}/{len(image_files)} images processed")
        logger.info(f"Total defects detected: {summary['total_defects']}")

        return summary

    def process_single_image(self,
                            image_path: str,
                            output_dir: Optional[str] = None,
                            save_annotated: bool = True) -> Dict:
        """
        Process a single image.

        Args:
            image_path: Path to the input image
            output_dir: Optional output directory
            save_annotated: Whether to save annotated image

        Returns:
            Dictionary with detection results
        """
        logger.info(f"Processing image: {image_path}")

        # Load and preprocess
        original, preprocessed, img_metadata = self.preprocessor.load_and_preprocess(
            image_path
        )

        # Detect defects
        defects = self.detector.detect_defects(preprocessed)

        logger.info(f"Detected {len(defects)} defects")

        # Save annotated image if requested
        annotated_path = None
        if save_annotated and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            image_name = Path(image_path).name
            annotated_path = output_path / f"annotated_{image_name}"

            self.visualizer.save_annotated_image(
                original, defects, str(annotated_path)
            )

        return {
            'status': 'success',
            'image_path': image_path,
            'defect_count': len(defects),
            'defects': defects,
            'metadata': img_metadata,
            'annotated_path': str(annotated_path) if annotated_path else None
        }

    def _find_image_files(self, directory: Path) -> List[Path]:
        """
        Find all supported image files in directory.

        Args:
            directory: Directory to search

        Returns:
            List of image file paths
        """
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))

        return sorted(image_files)


def create_batch_processor(config: Dict = None) -> BatchProcessor:
    """
    Factory function to create a configured batch processor.

    Args:
        config: Configuration dictionary

    Returns:
        Configured BatchProcessor instance
    """
    config = config or {}

    # Create components based on config
    detector_config = config.get('detector', {})
    detector = EnsembleDetector(detector_config) if config.get('use_ensemble', True) else DefectDetector(detector_config)

    preprocessor_config = config.get('preprocessor', {})
    preprocessor = ImagePreprocessor(
        target_size=preprocessor_config.get('target_size', None)
    )

    viz_mode = config.get('visualization_mode', 'both')
    visualizer = DefectVisualizer(visualization_mode=viz_mode)

    return BatchProcessor(detector, preprocessor, visualizer)
