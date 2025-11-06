"""
Defect Detection Module - SIMPLIFIED & BEGINNER-FRIENDLY VERSION

This file detects defects in microscope images using 4 simple methods.
Each method is explained with lots of comments!

WHAT IT DOES:
1. Finds bright and dark spots (particles and voids)
2. Finds scratches and edges
3. Finds specific particles and voids
4. Finds texture problems

HOW TO USE:
    detector = DefectDetector()
    defects = detector.detect_defects(image)
"""

import cv2
import numpy as np


# ============================================================================
# DEFECT CLASS - Stores information about one defect
# ============================================================================

class Defect:
    """
    A simple container for defect information.

    Like a box that holds:
    - bbox: Where the defect is (x, y, width, height)
    - contour: The exact shape of the defect
    - area: How big it is (in pixels)
    - defect_type: What kind (scratch, void, particle, etc.)
    - confidence: How sure we are (0.0 to 1.0, where 1.0 is 100% sure)
    """
    def __init__(self, bbox, contour, area, defect_type, confidence):
        self.bbox = bbox
        self.contour = contour
        self.area = area
        self.defect_type = defect_type
        self.confidence = confidence


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class DefectDetector:
    """
    Finds defects in microscope images.

    BEGINNER TIP: This class uses 4 different ways to find defects,
    then combines the results. This is better than using just one method!
    """

    def __init__(self, min_area=10, max_area=50000):
        """
        Set up the detector.

        Parameters:
            min_area: Smallest defect to find (pixels). Smaller = more sensitive
            max_area: Largest defect to find (pixels). Prevents finding whole image!
        """
        # Save the size limits
        self.min_area = min_area
        self.max_area = max_area

        # Image processing settings (you can change these!)
        self.blur_size = 5           # Blur amount (must be odd number)
        self.adaptive_block = 11     # Neighborhood size (must be odd)
        self.adaptive_c = 2          # Threshold adjustment
        self.canny_low = 50          # Edge detection sensitivity (lower)
        self.canny_high = 150        # Edge detection sensitivity (upper)
        self.morph_size = 3          # Cleanup shape size


    def detect_defects(self, image):
        """
        MAIN FUNCTION - Finds all defects in an image.

        Parameters:
            image: Your microscope image (color or grayscale)

        Returns:
            A list of Defect objects

        HOW IT WORKS:
        1. Convert to grayscale (if needed)
        2. Run 4 different detection methods
        3. Combine results
        4. Remove duplicates
        """
        print("  Starting defect detection...")

        # STEP 1: Make sure image is grayscale
        if len(image.shape) == 3:  # If image has 3 channels (BGR color)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # STEP 2: Run all 4 detection methods
        all_defects = []

        print("    Method 1: Finding bright/dark spots...")
        defects1 = self.find_bright_dark_spots(gray)
        all_defects.extend(defects1)
        print(f"      Found {len(defects1)} defects")

        print("    Method 2: Finding edges and scratches...")
        defects2 = self.find_edges_and_scratches(gray)
        all_defects.extend(defects2)
        print(f"      Found {len(defects2)} defects")

        print("    Method 3: Finding voids and particles...")
        defects3 = self.find_voids_and_particles(gray)
        all_defects.extend(defects3)
        print(f"      Found {len(defects3)} defects")

        print("    Method 4: Finding texture problems...")
        defects4 = self.find_texture_problems(gray)
        all_defects.extend(defects4)
        print(f"      Found {len(defects4)} defects")

        # STEP 3: Remove duplicates
        print("    Removing duplicates...")
        final_defects = self.remove_duplicates(all_defects)
        print(f"  Total unique defects: {len(final_defects)}")

        return final_defects


    # ========================================================================
    # METHOD 1: Find bright and dark spots
    # ========================================================================

    def find_bright_dark_spots(self, gray_image):
        """
        Finds spots that are brighter or darker than their surroundings.

        GOOD FOR: Particles (bright) and voids (dark)

        HOW IT WORKS:
        - Uses "adaptive thresholding" which compares each pixel to nearby pixels
        - Good for images with uneven lighting
        """
        defects = []

        # Remove noise but keep edges sharp (bilateral filter is magic!)
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)

        # Adaptive threshold: Compare each pixel to its neighborhood
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,  # Max value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use weighted average
            cv2.THRESH_BINARY_INV,  # Invert (white defects on black)
            self.adaptive_block,  # Neighborhood size
            self.adaptive_c  # Subtract this from threshold
        )

        # Clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove dots

        # Find all the white regions (contours)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check each contour
        for contour in contours:
            area = cv2.contourArea(contour)

            # Only keep if size is in range
            if self.min_area <= area <= self.max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Create defect object
                defect = Defect(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    defect_type='threshold_anomaly',
                    confidence=0.8
                )
                defects.append(defect)

        return defects


    # ========================================================================
    # METHOD 2: Find edges and scratches
    # ========================================================================

    def find_edges_and_scratches(self, gray_image):
        """
        Finds defects by looking for edges.

        GOOD FOR: Scratches (long thin defects) and irregular edges

        HOW IT WORKS:
        - Uses "Canny edge detection" to find all edges
        - Connects nearby edges together
        - Long thin defects are probably scratches!
        """
        defects = []

        # Blur first to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (self.blur_size, self.blur_size), 0)

        # Find edges (Canny is a famous edge detector)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Make edges thicker so nearby edges connect
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find all edge shapes
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate aspect ratio (how elongated is it?)
                aspect_ratio = max(w, h) / (min(w, h) + 0.001)  # Add 0.001 to avoid divide by zero

                # If very elongated (ratio > 2), probably a scratch
                if aspect_ratio > 2:
                    defect_type = 'scratch'
                    confidence = 0.85
                else:
                    defect_type = 'edge_defect'
                    confidence = 0.75

                defect = Defect(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    defect_type=defect_type,
                    confidence=confidence
                )
                defects.append(defect)

        return defects


    # ========================================================================
    # METHOD 3: Find voids and particles specifically
    # ========================================================================

    def find_voids_and_particles(self, gray_image):
        """
        Specifically finds particles (bright spots) and voids (dark holes).

        GOOD FOR: Identifying particle vs void

        HOW IT WORKS:
        - Uses "morphological operations" (top-hat and black-hat)
        - Top-hat finds bright spots
        - Black-hat finds dark spots
        - Checks average brightness to classify
        """
        defects = []

        # Create a circular shape for morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        # Top-hat: Finds bright spots on darker background
        tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)

        # Black-hat: Finds dark spots on brighter background
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

        # Threshold both (Otsu finds best threshold automatically)
        _, tophat_thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, blackhat_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine both results
        combined = cv2.bitwise_or(tophat_thresh, blackhat_thresh)

        # Find shapes
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Figure out if it's a void or particle by checking brightness
                # Create a mask for this defect
                mask = np.zeros(gray_image.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                # Calculate average brightness of defect
                mean_brightness = cv2.mean(gray_image, mask=mask)[0]

                # Compare to whole image average
                if mean_brightness < np.mean(gray_image):
                    defect_type = 'void'  # Darker than average
                else:
                    defect_type = 'particle'  # Brighter than average

                defect = Defect(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    defect_type=defect_type,
                    confidence=0.8
                )
                defects.append(defect)

        return defects


    # ========================================================================
    # METHOD 4: Find texture problems
    # ========================================================================

    def find_texture_problems(self, gray_image):
        """
        Finds areas that look rough or textured differently.

        GOOD FOR: Surface roughness, unusual texture

        HOW IT WORKS:
        - Calculates "variance" (how much each area varies)
        - High variance = rough texture
        - Low variance = smooth texture
        - Finds areas that are unusually rough or smooth
        """
        defects = []

        # Calculate local variance using math: Var(X) = E[X²] - E[X]²

        # Step 1: Calculate average (mean) of image
        mean_blur = cv2.GaussianBlur(gray_image, (21, 21), 0)

        # Step 2: Calculate average of squared image
        gray_float = gray_image.astype(np.float32)
        squared = gray_float ** 2
        squared_blur = cv2.GaussianBlur(squared, (21, 21), 0)

        # Step 3: Variance = E[X²] - E[X]²
        mean_float = mean_blur.astype(np.float32)
        variance = squared_blur - (mean_float ** 2)
        variance = np.clip(variance, 0, None)  # No negative values

        # Step 4: Normalize to 0-255 range
        variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
        variance_norm = variance_norm.astype(np.uint8)

        # Threshold to find high variance regions
        _, variance_thresh = cv2.threshold(variance_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(variance_thresh, cv2.MORPH_CLOSE, kernel)

        # Find shapes
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                defect = Defect(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    defect_type='texture_anomaly',
                    confidence=0.7
                )
                defects.append(defect)

        return defects


    # ========================================================================
    # REMOVE DUPLICATES - Same defect found by multiple methods
    # ========================================================================

    def remove_duplicates(self, defects):
        """
        Removes duplicate detections.

        WHY: Multiple methods often find the same defect.
        We want to keep each defect only once!

        HOW IT WORKS:
        - Keeps defects with highest confidence first
        - If two defects overlap more than 30%, removes the lower confidence one
        """
        if not defects:
            return []

        # Sort by confidence (best first)
        defects = sorted(defects, key=lambda d: d.confidence, reverse=True)

        # List of defects to keep
        keep = []

        for defect in defects:
            # Check if this overlaps with any we're keeping
            is_duplicate = False

            for kept_defect in keep:
                # Calculate overlap
                overlap = self.calculate_overlap(defect.bbox, kept_defect.bbox)

                # If they overlap more than 30%, it's a duplicate
                if overlap > 0.3:
                    is_duplicate = True
                    break

            # If not duplicate, keep it!
            if not is_duplicate:
                keep.append(defect)

        return keep


    def calculate_overlap(self, bbox1, bbox2):
        """
        Calculates how much two bounding boxes overlap.

        Returns: Number from 0.0 (no overlap) to 1.0 (complete overlap)

        This is called "IoU" (Intersection over Union) in computer vision.

        EXAMPLE:
        If box1 and box2 overlap 50 pixels, and together they cover 100 pixels,
        IoU = 50/100 = 0.5
        """
        # Unpack bounding boxes
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Find the overlapping rectangle
        x_left = max(x1, x2)  # Leftmost point of overlap
        y_top = max(y1, y2)  # Topmost point of overlap
        x_right = min(x1 + w1, x2 + w2)  # Rightmost point of overlap
        y_bottom = min(y1 + h1, y2 + h2)  # Bottommost point of overlap

        # Check if there's no overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate overlap area
        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate total area covered by both boxes
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection  # Subtract intersection (counted twice)

        # Return IoU
        if union > 0:
            return intersection / union
        else:
            return 0.0


# ============================================================================
# ENSEMBLE DETECTOR - Runs detection twice for better results
# ============================================================================

class EnsembleDetector(DefectDetector):
    """
    Enhanced detector that runs detection TWICE with different settings.

    WHY: Running twice catches more defects!
    First pass uses normal settings.
    Second pass uses more sensitive settings.

    BEGINNER TIP: This is slower but more accurate. Use this for important work!
    """

    def detect_defects(self, image):
        """
        Run detection twice and combine results.
        """
        print("  Using ENSEMBLE detection (2 passes)...")

        all_defects = []

        # PASS 1: Normal settings
        print("  Pass 1: Normal sensitivity...")
        defects1 = super().detect_defects(image)
        all_defects.extend(defects1)

        # PASS 2: Higher sensitivity
        print("  Pass 2: High sensitivity...")

        # Save original settings
        original_min = self.min_area
        original_c = self.adaptive_c

        # Make more sensitive
        self.min_area = 5  # Catch smaller defects
        self.adaptive_c = 1  # More sensitive threshold

        defects2 = super().detect_defects(image)
        all_defects.extend(defects2)

        # Restore original settings
        self.min_area = original_min
        self.adaptive_c = original_c

        # Remove duplicates
        print("  Combining results...")
        final_defects = self.remove_duplicates(all_defects)

        return final_defects


# ============================================================================
# HELPER FUNCTION - Easy way to create detector
# ============================================================================

def create_detector(min_area=10, max_area=50000, use_ensemble=True):
    """
    Easy function to create a detector.

    Parameters:
        min_area: Minimum defect size (pixels)
        max_area: Maximum defect size (pixels)
        use_ensemble: True = more accurate, False = faster

    Returns:
        A ready-to-use detector

    EXAMPLE:
        detector = create_detector(min_area=15, use_ensemble=True)
        defects = detector.detect_defects(my_image)
    """
    if use_ensemble:
        return EnsembleDetector(min_area, max_area)
    else:
        return DefectDetector(min_area, max_area)
