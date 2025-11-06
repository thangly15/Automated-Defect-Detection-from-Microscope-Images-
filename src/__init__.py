"""
Automated Defect Detection System

A comprehensive defect detection system for microscope images.
"""

__version__ = "1.0.0"
__author__ = "Thang Ly"

from .defect_detector import DefectDetector, EnsembleDetector, Defect
from .image_preprocessor import ImagePreprocessor
from .visualizer import DefectVisualizer
from .report_generator import ReportGenerator, Statistics
from .batch_processor import BatchProcessor

__all__ = [
    'DefectDetector',
    'EnsembleDetector',
    'Defect',
    'ImagePreprocessor',
    'DefectVisualizer',
    'ReportGenerator',
    'Statistics',
    'BatchProcessor',
]
