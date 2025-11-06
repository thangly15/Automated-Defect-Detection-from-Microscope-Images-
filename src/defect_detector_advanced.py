"""
Defect Detection Module - SIMPLIFIED VERSION

This module detects defects in microscope images using 4 different methods:
1. Finding dark and bright spots (adaptive thresholding)
2. Finding edges and scratches (edge detection)
3. Finding voids and particles (morphological operations)
4. Finding texture problems (texture analysis)
"""

import cv2
import numpy as np


class Defect:
    """
    Simple class to store information about one defect.

    Think of this like a container that holds:
    - Where the defect is (bounding box)
    - What shape it is (contour)
    - How big it is (area)
    - What type it is (scratch, void, etc.)
    - How confident we are (confidence score)
    """
    def __init__(self, bbox, contour, area, defect_type, confidence):
        self.bbox = bbox              # (x, y, width, height)
        self.contour = contour        # The exact shape
        self.area = area              # Size in pixels
        self.defect_type = defect_type  # "scratch", "void", etc.
        self.confidence = confidence  # How sure we are (0.0 to 1.0)


class DefectDetector:
    """
    Main class that finds defects in images.

    It uses 4 different detection methods and combines them together.
    """

    def __init__(self, min_area=10, max_area=50000):
        """
        Initialize the detector.

        Parameters:
            min_area: Smallest defect size to detect (in pixels)
            max_area: Largest defect size to detect (in pixels)
        """
        # Store settings
        self.min_area = min_area
        self.max_area = max_area

        # Settings for image processing
        self.blur_size = 5              # How much to blur (reduce noise)
        self.adaptive_block = 11        # Size of local area for thresholding
        self.adaptive_c = 2             # Fine-tuning parameter
        self.canny_low = 50             # Edge detection - lower threshold
        self.canny_high = 150           # Edge detection - upper threshold
        self.morph_size = 3             # Size for morphological operations

    def detect_defects(self, image):
        """
        Main function to detect all defects in an image.

        Parameters:
            image: Input image (can be color or grayscale)

        Returns:
            List of Defect objects found in the image
        """
        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 2: Use all 4 detection methods
        all_defects = []

        # Method 1: Find bright/dark spots
        defects1 = self.find_bright_dark_spots(gray)
        all_defects.extend(defects1)

        # Method 2: Find edges and scratches
        defects2 = self.find_edges_and_scratches(gray)
        all_defects.extend(defects2)

        # Method 3: Find voids and particles
        defects3 = self.find_voids_and_particles(gray)
        all_defects.extend(defects3)

        # Method 4: Find texture problems
        defects4 = self.find_texture_problems(gray)
        all_defects.extend(defects4)

        # Step 3: Remove duplicates (same defect found by multiple methods)
        final_defects = self.remove_duplicates(all_defects)

        return final_defects

    def find_bright_dark_spots(self, gray_image):
        """
        METHOD 1: Find defects that are brighter or darker than surroundings.

        This catches particles (bright spots) and voids (dark spots).
        """
        defects = []

        # Reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)

        # Adaptive threshold: each pixel compared to its neighbors
        # This handles images with uneven lighting
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block,
            self.adaptive_c
        )

        # Clean up the result
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find the shapes (contours)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check each contour
        for contour in contours:
            area = cv2.contourArea(contour)

            # Only keep if size is in our range
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Create a defect object
                defect = Defect(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    defect_type='threshold_anomaly',
                    confidence=0.8
                )
                defects.append(defect)

        return defects

    def _detect_edge_defects(self, gray: np.ndarray) -> List[Defect]:
        """Detect scratches and irregular edges using Canny edge detection"""
        defects = []

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray,
                                   (self.config['gaussian_kernel'],
                                    self.config['gaussian_kernel']), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred,
                         self.config['canny_low'],
                         self.config['canny_high'])

        # Dilate edges to connect nearby edge pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.config['min_defect_area']:
                # Check if it's elongated (typical for scratches)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

                if aspect_ratio > 2:  # Likely a scratch
                    defects.append(Defect(
                        bbox=(x, y, w, h),
                        contour=contour,
                        area=area,
                        defect_type='scratch',
                        confidence=0.85 * self.config['edge_sensitivity']
                    ))
                else:
                    defects.append(Defect(
                        bbox=(x, y, w, h),
                        contour=contour,
                        area=area,
                        defect_type='edge_defect',
                        confidence=0.75 * self.config['edge_sensitivity']
                    ))

        return defects

    def _detect_morphological_defects(self, gray: np.ndarray) -> List[Defect]:
        """Detect voids and particles using morphological operations"""
        defects = []

        # Top-hat transform to detect bright particles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Black-hat transform to detect dark voids
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold the transforms
        _, tophat_thresh = cv2.threshold(tophat, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, blackhat_thresh = cv2.threshold(blackhat, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine both
        combined = cv2.bitwise_or(tophat_thresh, blackhat_thresh)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_defect_area'] <= area <= self.config['max_defect_area']:
                x, y, w, h = cv2.boundingRect(contour)

                # Determine if it's a void or particle based on intensity
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]

                if mean_intensity < np.mean(gray):
                    defect_type = 'void'
                else:
                    defect_type = 'particle'

                defects.append(Defect(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    defect_type=defect_type,
                    confidence=0.8
                ))

        return defects

    def _detect_texture_anomalies(self, gray: np.ndarray) -> List[Defect]:
        """Detect surface texture anomalies using variance analysis"""
        defects = []

        # Calculate local variance
        mean_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        sqr_blur = cv2.GaussianBlur(gray.astype(np.float32)**2, (21, 21), 0)
        variance = sqr_blur - mean_blur.astype(np.float32)**2
        variance = np.clip(variance, 0, None)

        # Normalize variance
        variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
        variance_norm = variance_norm.astype(np.uint8)

        # Threshold high variance regions
        _, variance_thresh = cv2.threshold(variance_norm, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(variance_thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_defect_area'] <= area <= self.config['max_defect_area']:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append(Defect(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    defect_type='texture_anomaly',
                    confidence=0.7 * self.config['texture_sensitivity']
                ))

        return defects

    def _remove_duplicates(self, defects: List[Defect], iou_threshold: float = 0.3) -> List[Defect]:
        """
        Remove duplicate detections using Non-Maximum Suppression.

        Args:
            defects: List of detected defects
            iou_threshold: IoU threshold for considering defects as duplicates

        Returns:
            Filtered list of defects
        """
        if not defects:
            return []

        # Sort by confidence (descending)
        defects = sorted(defects, key=lambda d: d.confidence, reverse=True)

        keep = []
        for i, defect in enumerate(defects):
            should_keep = True
            for kept_defect in keep:
                if self._calculate_iou(defect.bbox, kept_defect.bbox) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(defect)

        return keep

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class EnsembleDetector(DefectDetector):
    """
    Enhanced detector with ensemble voting for increased reliability.
    """

    def detect_defects(self, image: np.ndarray) -> List[Defect]:
        """
        Enhanced detection with multiple parameter sets and voting.
        """
        all_defects = []

        # Run detection with default parameters
        all_defects.extend(super().detect_defects(image))

        # Run detection with higher sensitivity
        original_config = self.config.copy()
        self.config['min_defect_area'] = 5
        self.config['adaptive_c'] = 1
        all_defects.extend(super().detect_defects(image))

        # Restore original config
        self.config = original_config

        # Remove duplicates with stricter IoU
        return self._remove_duplicates(all_defects, iou_threshold=0.5)
