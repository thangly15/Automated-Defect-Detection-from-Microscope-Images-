# Automated Defect Detection System - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command-Line Interface](#command-line-interface)
4. [Web Interface](#web-interface)
5. [Configuration](#configuration)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- OpenCV for image processing
- NumPy for numerical operations
- Pandas for data handling
- scikit-image for advanced image processing
- tqdm for progress bars
- PyYAML for configuration
- Streamlit for web interface (optional)

---

## Quick Start

### Process a Single Image

```bash
python detect_defects.py --input path/to/image.jpg --output results/
```

### Process Multiple Images

```bash
python detect_defects.py --input path/to/images/ --output results/
```

### Launch Web Interface

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

---

## Command-Line Interface

### Basic Usage

```bash
python detect_defects.py --input <input_path> --output <output_path> [options]
```

### Required Arguments

- `-i, --input`: Input image file or directory
- `-o, --output`: Output directory for results

### Optional Arguments

- `--config PATH`: Path to configuration YAML file
- `--ensemble`: Use ensemble detection (default: True)
- `--no-ensemble`: Disable ensemble detection (faster but less accurate)
- `--min-area FLOAT`: Minimum defect area in pixels (default: 10)
- `--max-area FLOAT`: Maximum defect area in pixels (default: 50000)
- `--visualization MODE`: Visualization mode (bbox, contour, both, heatmap)
- `--no-reports`: Skip generating CSV/JSON reports
- `--no-annotated`: Skip saving annotated images
- `--verbose`: Enable verbose logging

### Examples

#### Basic detection with default settings
```bash
python detect_defects.py -i microscope_image.jpg -o results/
```

#### Batch processing with custom parameters
```bash
python detect_defects.py \
  --input images/ \
  --output results/ \
  --min-area 20 \
  --max-area 30000 \
  --visualization heatmap
```

#### Using a configuration file
```bash
python detect_defects.py \
  --input images/ \
  --output results/ \
  --config my_config.yaml
```

#### Fast processing (no ensemble)
```bash
python detect_defects.py \
  --input images/ \
  --output results/ \
  --no-ensemble
```

---

## Web Interface

### Starting the Web Interface

```bash
streamlit run streamlit_app.py
```

### Features

1. **Upload & Detect**: Upload single images for analysis
2. **Interactive Configuration**: Adjust detection parameters in real-time
3. **Visual Results**: View original and annotated images side-by-side
4. **Statistics Dashboard**: See defect counts, types, and quality metrics
5. **Download Results**: Download annotated images

### Using the Web Interface

1. Open browser to `http://localhost:8501`
2. Configure detection parameters in sidebar (optional)
3. Upload an image using the file uploader
4. Click "Detect Defects" button
5. Review results and download annotated image

---

## Configuration

### Configuration File Format

Create a YAML file (e.g., `config.yaml`) with the following structure:

```yaml
use_ensemble: true
visualization_mode: both

detector:
  min_defect_area: 10
  max_defect_area: 50000
  edge_sensitivity: 0.7
  texture_sensitivity: 0.6
```

See `config.yaml` for a complete example with all available parameters.

### Key Parameters

#### Detection Sensitivity

- `min_defect_area`: Smaller values detect more defects (including noise)
- `max_defect_area`: Prevents detecting very large regions
- `edge_sensitivity` (0.0-1.0): Higher = more sensitive to edges
- `texture_sensitivity` (0.0-1.0): Higher = more sensitive to texture anomalies

#### Preprocessing

- `gaussian_kernel`: Size of Gaussian blur kernel (odd number)
- `adaptive_block_size`: Size of neighborhood for adaptive thresholding
- `canny_low/high`: Thresholds for Canny edge detection

#### Advanced Tuning

For specific defect types or image characteristics, adjust advanced parameters:

- `scratch.min_aspect_ratio`: Minimum elongation for scratch detection
- `tophat_kernel_size`: Kernel size for particle detection
- `texture_window_size`: Window size for texture analysis

---

## Output Files

### Directory Structure

```
output/
├── annotated/              # Annotated images
│   ├── annotated_image1.jpg
│   ├── annotated_image2.jpg
│   └── ...
├── defect_report.csv       # Detailed defect data
├── defect_summary.csv      # Summary counts per image
├── defect_report.json      # Structured JSON report
└── summary.txt             # Human-readable summary
```

### CSV Report Format

**defect_report.csv** - Detailed information about each defect:
```csv
image_name,defect_id,defect_type,bbox_x,bbox_y,bbox_width,bbox_height,area,confidence
image1.jpg,1,scratch,245,123,45,180,1250.5,0.85
image1.jpg,2,void,567,890,35,35,962.3,0.80
...
```

**defect_summary.csv** - Summary per image:
```csv
image_name,total_defects,scratches,voids,particles,edge_defects,texture_anomalies,threshold_anomalies
image1.jpg,15,3,2,5,3,1,1
image2.jpg,8,1,0,4,2,1,0
...
```

### JSON Report Format

Structured JSON with detailed information:

```json
{
  "generated_at": "2024-01-15T10:30:00",
  "total_images": 10,
  "total_defects": 127,
  "images": [
    {
      "filename": "image1.jpg",
      "defect_count": 15,
      "metadata": {
        "focus_quality": 0.85,
        "noise_level": 0.12,
        "contrast": 0.67
      },
      "defects": [
        {
          "id": 1,
          "type": "scratch",
          "bounding_box": {"x": 245, "y": 123, "width": 45, "height": 180},
          "area": 1250.5,
          "confidence": 0.85
        }
      ]
    }
  ]
}
```

### Text Summary

Human-readable summary with statistics:
- Overall defect counts
- Defects by type
- Per-image breakdown

---

## Troubleshooting

### Common Issues

#### 1. Too Many False Positives

**Problem**: System detects noise or artifacts as defects

**Solutions**:
- Increase `min_defect_area` (try 20-30 pixels)
- Check image quality metrics (focus, noise level)
- Reduce sensitivity parameters
- Use ensemble detection

#### 2. Missing Defects

**Problem**: System fails to detect visible defects

**Solutions**:
- Decrease `min_defect_area` (try 5-10 pixels)
- Enable ensemble detection (`--ensemble`)
- Check if defects are within area range
- Adjust sensitivity parameters upward
- Review image quality (ensure good focus and lighting)

#### 3. Slow Processing

**Problem**: Processing takes too long

**Solutions**:
- Disable ensemble detection (`--no-ensemble`)
- Resize images to smaller resolution
- Increase `min_defect_area` to reduce processing
- Process smaller batches

#### 4. Poor Image Quality

**Problem**: Low focus quality or high noise level reported

**Solutions**:
- Use higher resolution images
- Improve microscope focus
- Ensure consistent lighting
- Reduce camera noise

### Error Messages

#### "Failed to load image"
- Check file path is correct
- Verify image format is supported (JPG, PNG, TIFF, BMP)
- Ensure file is not corrupted

#### "No image files found"
- Verify directory path
- Check file extensions
- Ensure images are in top-level directory (not subdirectories)

---

## Advanced Usage

### Python API

Use the system programmatically in your own Python scripts:

```python
from src.batch_processor import BatchProcessor
from src.defect_detector import EnsembleDetector
from src.image_preprocessor import ImagePreprocessor
from src.visualizer import DefectVisualizer

# Create components
detector = EnsembleDetector()
preprocessor = ImagePreprocessor()
visualizer = DefectVisualizer(visualization_mode='both')

# Create processor
processor = BatchProcessor(detector, preprocessor, visualizer)

# Process single image
result = processor.process_single_image(
    'path/to/image.jpg',
    'output/',
    save_annotated=True
)

print(f"Detected {result['defect_count']} defects")

# Process directory
summary = processor.process_directory(
    'input_images/',
    'output/',
    generate_reports=True
)

print(f"Processed {summary['processed_images']} images")
print(f"Total defects: {summary['total_defects']}")
```

### Custom Configuration

Create a custom detector with specific parameters:

```python
from src.defect_detector import DefectDetector

config = {
    'min_defect_area': 20,
    'max_defect_area': 30000,
    'edge_sensitivity': 0.8,
    'texture_sensitivity': 0.5,
    'adaptive_block_size': 15,
    'canny_low': 40,
    'canny_high': 120
}

detector = DefectDetector(config)
defects = detector.detect_defects(image)
```

### Extending the System

#### Add Custom Defect Type

```python
from src.defect_detector import DefectDetector, Defect

class CustomDetector(DefectDetector):
    def detect_defects(self, image):
        # Get base detections
        defects = super().detect_defects(image)

        # Add custom detection logic
        custom_defects = self._detect_custom_defects(image)
        defects.extend(custom_defects)

        return self._remove_duplicates(defects)

    def _detect_custom_defects(self, image):
        # Your custom detection logic here
        defects = []
        # ... detection code ...
        return defects
```

### Integration with Other Tools

#### Export to YOLO Format

```python
# Convert detections to YOLO format for training
def export_yolo_format(image_path, defects, output_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(output_path, 'w') as f:
        for defect in defects:
            x, y, box_w, box_h = defect.bbox

            # Convert to YOLO format (normalized center coordinates)
            x_center = (x + box_w/2) / w
            y_center = (y + box_h/2) / h
            width = box_w / w
            height = box_h / h

            class_id = 0  # Map defect type to class ID
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
```

---

## Performance Tips

1. **Image Size**: Resize very large images for faster processing
2. **Ensemble vs Single**: Use single detector for speed, ensemble for accuracy
3. **Batch Processing**: Process multiple images at once using CLI
4. **Parameter Tuning**: Start with defaults, adjust based on results
5. **Hardware**: More RAM allows processing larger images

---

## Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

### Resources

- Project documentation: `README.md`
- Configuration reference: `config.yaml`
- Source code: `src/` directory
- Example outputs: `examples/` directory

---

## Version History

### v1.0.0
- Initial release
- Multi-strategy defect detection
- CLI and web interfaces
- CSV/JSON report generation
- Ensemble detection support
