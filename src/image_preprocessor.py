"""
Image Preprocessing Module

Handles image loading, normalization, and enhancement for microscope images.
Addresses challenges like variable lighting, focus conditions, and noise.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ImagePreprocessor:
    """
    Preprocesses microscope images for defect detection.

    Handles:
    - Image loading from various formats
    - Lighting normalization
    - Noise reduction
    - Contrast enhancement
    - Focus quality assessment
    """

    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the preprocessor.

        Args:
            target_size: Optional (width, height) to resize images to
        """
        self.target_size = target_size

    def load_and_preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load and preprocess an image for defect detection.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (original_image, preprocessed_image, metadata)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original = image.copy()

        # Resize if needed
        if self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Calculate image quality metrics
        metadata = self._calculate_metadata(image, image_path)

        # Apply preprocessing pipeline
        preprocessed = self._preprocess_pipeline(image, metadata)

        return original, preprocessed, metadata

    def _preprocess_pipeline(self, image: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Apply preprocessing operations to enhance defect visibility.

        Args:
            image: Input image
            metadata: Image metadata for adaptive processing

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed (many operations work better on grayscale)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Lighting normalization using CLAHE
        preprocessed = self._normalize_lighting(gray)

        # 2. Noise reduction while preserving edges
        preprocessed = self._denoise(preprocessed, metadata['noise_level'])

        # 3. Enhance contrast adaptively
        preprocessed = self._enhance_contrast(preprocessed)

        # 4. Sharpening (if image is slightly out of focus)
        if metadata['focus_quality'] < 0.7:
            preprocessed = self._sharpen(preprocessed)

        return preprocessed

    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize lighting using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        This helps with variable lighting conditions across the image.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(image)
        return normalized

    def _denoise(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Apply adaptive denoising based on estimated noise level.

        Args:
            image: Input image
            noise_level: Estimated noise level (0-1)

        Returns:
            Denoised image
        """
        if noise_level > 0.3:
            # High noise: stronger denoising
            denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7,
                                               searchWindowSize=21)
        elif noise_level > 0.15:
            # Medium noise: moderate denoising
            denoised = cv2.fastNlMeansDenoising(image, h=6, templateWindowSize=7,
                                               searchWindowSize=21)
        else:
            # Low noise: light denoising
            denoised = cv2.GaussianBlur(image, (3, 3), 0)

        return denoised

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using histogram stretching.
        """
        # Calculate percentiles for robust normalization
        p2, p98 = np.percentile(image, (2, 98))

        # Stretch histogram
        if p98 > p2:
            enhanced = np.clip((image - p2) * 255.0 / (p98 - p2), 0, 255)
            enhanced = enhanced.astype(np.uint8)
        else:
            enhanced = image

        return enhanced

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp masking.
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened

    def _calculate_metadata(self, image: np.ndarray, image_path: str) -> dict:
        """
        Calculate image quality metrics and metadata.

        Args:
            image: Input image
            image_path: Path to the image file

        Returns:
            Dictionary with metadata
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        metadata = {
            'path': image_path,
            'filename': Path(image_path).name,
            'shape': image.shape,
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'focus_quality': self._estimate_focus_quality(gray),
            'noise_level': self._estimate_noise_level(gray),
            'contrast': self._estimate_contrast(gray)
        }

        return metadata

    def _estimate_focus_quality(self, gray: np.ndarray) -> float:
        """
        Estimate focus quality using Laplacian variance.

        Higher values indicate better focus.
        Returns normalized score between 0 and 1.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to 0-1 range (empirically determined thresholds for microscope images)
        focus_score = min(variance / 500.0, 1.0)
        return focus_score

    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """
        Estimate noise level using high-frequency content analysis.

        Returns normalized noise level between 0 and 1.
        """
        # Apply median filter
        median = cv2.medianBlur(gray, 5)

        # Calculate difference (noise estimate)
        noise = cv2.absdiff(gray, median)
        noise_level = np.std(noise) / 255.0

        return noise_level

    def _estimate_contrast(self, gray: np.ndarray) -> float:
        """
        Estimate image contrast using Michelson contrast.

        Returns normalized contrast score between 0 and 1.
        """
        max_intensity = np.max(gray)
        min_intensity = np.min(gray)

        if max_intensity + min_intensity > 0:
            contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
        else:
            contrast = 0.0

        return contrast


def create_preprocessor(config: dict = None) -> ImagePreprocessor:
    """
    Factory function to create a preprocessor with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured ImagePreprocessor instance
    """
    config = config or {}
    target_size = config.get('target_size', None)
    return ImagePreprocessor(target_size=target_size)
