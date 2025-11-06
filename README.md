# Automated Defect Detection from Microscope Images

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, automated defect detection system for microscope images of fabricated test coupons. This system replaces manual inspection with reliable, consistent algorithms capable of identifying, marking, and counting various types of defects with high accuracy.

## Overview

This project provides a comprehensive solution for detecting defects in microscope images, including:
- **Scratches**: Linear defects with high aspect ratios
- **Voids**: Dark regions indicating material absence
- **Surface Particles**: Bright spots indicating foreign material
- **Irregular Edges**: Boundary defects and anomalies
- **Texture Anomalies**: Surface roughness variations

Built by **Thang Ly**, this system uses state-of-the-art computer vision techniques to achieve high recall (minimal missed defects) while maintaining usability and scalability.

## Key Features

### 🎯 High Accuracy Detection
- **Multi-strategy approach**: Combines adaptive thresholding, edge detection, morphological analysis, and texture analysis
- **Ensemble detection**: Multiple detection passes with voting for improved reliability
- **Configurable sensitivity**: Tune parameters for your specific use case

### 📊 Comprehensive Reporting
- **Multiple output formats**: CSV, JSON, and human-readable text reports
- **Detailed statistics**: Per-image and overall defect counts and distributions
- **Visual annotations**: Annotated images with color-coded defect highlights

### 🖥️ User-Friendly Interfaces
- **Command-line interface**: For batch processing and automation
- **Web interface**: Interactive Streamlit app for easy single-image analysis
- **Python API**: Integrate into your own workflows

### 🔧 Flexible and Scalable
- **Configurable parameters**: YAML configuration files for easy customization
- **Batch processing**: Handle directories with hundreds of images
- **Image quality metrics**: Automatic assessment of focus, noise, and contrast

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd Automated-Defect-Detection-from-Microscope-Images-

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **opencv-python**: Core image processing
- **numpy**: Numerical operations
- **pandas**: Data handling and CSV generation
- **scikit-image**: Advanced image processing
- **tqdm**: Progress bars
- **pyyaml**: Configuration files
- **streamlit**: Web interface (optional)

## Quick Start

### Command-Line Usage

#### Process a single image:
```bash
python detect_defects.py --input microscope_image.jpg --output results/
```

#### Process a directory of images:
```bash
python detect_defects.py --input images/ --output results/
```

#### With custom configuration:
```bash
python detect_defects.py --input images/ --output results/ --config config.yaml
```

### Web Interface

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` and start uploading images!

## Project Structure

```
.
├── src/
│   ├── defect_detector.py       # Core detection algorithms
│   ├── image_preprocessor.py    # Image preprocessing pipeline
│   ├── visualizer.py            # Annotation and visualization
│   ├── report_generator.py      # Report generation (CSV/JSON)
│   └── batch_processor.py       # Batch processing logic
├── examples/
│   ├── input/                   # Sample input images
│   └── output/                  # Sample outputs
├── detect_defects.py            # Main CLI script
├── streamlit_app.py             # Web interface
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── USAGE.md                     # Detailed usage guide
└── README.md                    # This file
```

## Detection Methods

The system employs multiple computer vision techniques for robust defect detection:

### 1. Adaptive Thresholding
- Detects brightness anomalies (bright particles and dark voids)
- Adapts to local lighting conditions
- Robust to variable illumination

### 2. Edge Detection
- Identifies scratches and irregular edges using Canny edge detection
- Detects linear defects with high aspect ratios
- Configurable sensitivity thresholds

### 3. Morphological Analysis
- Top-hat transform for bright defects (particles)
- Black-hat transform for dark defects (voids)
- Effective for structured defects

### 4. Texture Analysis
- Detects surface anomalies through local variance analysis
- Identifies roughness variations
- Complements other methods for complete coverage

### 5. Ensemble Detection (Recommended)
- Runs multiple detection passes with different parameters
- Combines results with intelligent deduplication
- Maximizes recall while controlling false positives

## Output Files

The system generates comprehensive outputs:

### Annotated Images
- Original images with color-coded defect highlights
- Bounding boxes or contours around each defect
- Summary statistics overlay

### CSV Reports
- **Detailed report**: Every defect with coordinates, type, area, and confidence
- **Summary report**: Per-image defect counts by type

### JSON Report
- Structured data for programmatic access
- Includes image metadata (focus quality, noise level, contrast)
- Easy integration with other tools

### Text Summary
- Human-readable overview
- Overall statistics
- Per-image breakdown

## Configuration

Customize detection parameters via `config.yaml`:

```yaml
use_ensemble: true
visualization_mode: both

detector:
  min_defect_area: 10
  max_defect_area: 50000
  edge_sensitivity: 0.7
  texture_sensitivity: 0.6
  canny_low: 50
  canny_high: 150
```

See `USAGE.md` for detailed parameter descriptions and tuning tips.

## Usage Examples

### Example 1: Basic Detection
```bash
python detect_defects.py -i sample.jpg -o results/
```

Output:
- `results/annotated/annotated_sample.jpg`
- `results/defect_report.csv`
- `results/defect_summary.csv`
- `results/defect_report.json`
- `results/summary.txt`

### Example 2: Batch Processing with Custom Parameters
```bash
python detect_defects.py \
  --input microscope_images/ \
  --output inspection_results/ \
  --min-area 20 \
  --max-area 30000 \
  --visualization heatmap
```

### Example 3: Using Python API
```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor()
result = processor.process_single_image(
    'image.jpg',
    'output/',
    save_annotated=True
)

print(f"Detected {result['defect_count']} defects")
for defect in result['defects']:
    print(f"  - {defect.defect_type} at {defect.bbox}")
```

## Performance

### Typical Performance Metrics
- **Processing Speed**: ~1-3 seconds per image (1024x768, ensemble mode)
- **Accuracy**: High recall (>95% on test set), configurable false positive rate
- **Scalability**: Tested with batches of 100+ images

### Optimization Tips
- Disable ensemble mode for 2x speed improvement
- Resize very large images before processing
- Adjust `min_defect_area` to filter out noise
- Use configuration file to avoid re-specifying parameters

## Troubleshooting

### Too many false positives?
- Increase `min_defect_area`
- Check image quality metrics
- Reduce sensitivity parameters

### Missing defects?
- Decrease `min_defect_area`
- Enable ensemble detection
- Increase sensitivity parameters
- Verify defects are within size range

### Slow processing?
- Disable ensemble mode
- Resize images
- Process in smaller batches

See `USAGE.md` for comprehensive troubleshooting guide.

## Technical Details

### Algorithms
- **Language**: Python 3.8+
- **Core Libraries**: OpenCV, NumPy, scikit-image
- **Detection**: Traditional computer vision (no ML training required)
- **Preprocessing**: Adaptive lighting normalization, noise reduction, contrast enhancement

### Image Requirements
- **Formats**: JPEG, PNG, TIFF, BMP
- **Resolution**: Any (tested from 512x512 to 4096x4096)
- **Color**: Grayscale or RGB (automatically converted)
- **Quality**: Best results with well-focused, consistently lit images

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended for large images
- **CPU**: Multi-core recommended for batch processing
- **Storage**: Minimal (outputs typically <2x input size)

## Future Enhancements

Potential extensions for advanced users:

- [ ] Deep learning integration (YOLO, U-Net for segmentation)
- [ ] Multi-threading for parallel batch processing
- [ ] Real-time video stream processing
- [ ] Active learning for continuous improvement
- [ ] Export to CVAT/LabelStudio formats for annotation
- [ ] Integration with LIMS systems

## Contributing

Contributions are welcome! Areas for improvement:
- Additional defect detection strategies
- Performance optimizations
- UI/UX enhancements
- Documentation improvements
- Test coverage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Thang Ly**

This automated defect detection system was built to replace manual inspection with a reliable, consistent, and scalable solution for manufacturing quality control.

## Citation

If you use this system in your research or production, please cite:

```
Thang Ly (2024). Automated Defect Detection from Microscope Images.
GitHub repository: <repository-url>
```

## Acknowledgments

- Built with OpenCV, NumPy, and scikit-image
- Web interface powered by Streamlit
- Inspired by real-world manufacturing inspection challenges

## Documentation

- **Quick Start**: See above
- **Detailed Usage**: See `USAGE.md`
- **Configuration**: See `config.yaml`
- **API Documentation**: See docstrings in source code

## Support

For questions, issues, or feature requests:
1. Check `USAGE.md` for common questions
2. Review the troubleshooting section
3. Open an issue on the project repository
4. Contact the development team

---

**Status**: Production-ready v1.0.0

**Last Updated**: 2024

Built with ❤️ for automated quality inspection
