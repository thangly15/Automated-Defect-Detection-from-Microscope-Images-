"""
Streamlit Web Interface for Defect Detection System

A user-friendly web interface for uploading and analyzing microscope images.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import pandas as pd
from PIL import Image
import io

from src.batch_processor import BatchProcessor
from src.defect_detector import EnsembleDetector, DefectDetector
from src.image_preprocessor import ImagePreprocessor
from src.visualizer import DefectVisualizer
from src.report_generator import Statistics


# Page configuration
st.set_page_config(
    page_title="Defect Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_processor(use_ensemble: bool, min_area: float, max_area: float, viz_mode: str):
    """Load and cache the processor"""
    config = {
        'use_ensemble': use_ensemble,
        'visualization_mode': viz_mode,
        'detector': {
            'min_defect_area': min_area,
            'max_defect_area': max_area
        }
    }

    detector_config = config.get('detector', {})
    detector = EnsembleDetector(detector_config) if use_ensemble else DefectDetector(detector_config)

    preprocessor = ImagePreprocessor()
    visualizer = DefectVisualizer(visualization_mode=viz_mode)

    return BatchProcessor(detector, preprocessor, visualizer)


def process_uploaded_file(uploaded_file, processor):
    """Process an uploaded image file"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_output:
        result = processor.process_single_image(
            tmp_path,
            tmp_output,
            save_annotated=True
        )

        # Load annotated image
        annotated_img = cv2.imread(result['annotated_path'])
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Clean up
        Path(tmp_path).unlink()

    return result, annotated_img


def display_defect_table(defects):
    """Display defects in a table format"""
    if not defects:
        st.info("No defects detected")
        return

    data = []
    for i, defect in enumerate(defects, 1):
        x, y, w, h = defect.bbox
        data.append({
            'ID': i,
            'Type': defect.defect_type,
            'X': x,
            'Y': y,
            'Width': w,
            'Height': h,
            'Area (px²)': f"{defect.area:.1f}",
            'Confidence': f"{defect.confidence:.2f}"
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Automated Defect Detection System</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Upload microscope images to automatically detect and analyze defects including:
    scratches, voids, surface particles, and irregular edges.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    use_ensemble = st.sidebar.checkbox(
        "Use Ensemble Detection",
        value=True,
        help="Use multiple detection strategies for higher accuracy (recommended)"
    )

    min_area = st.sidebar.slider(
        "Minimum Defect Area (px²)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Minimum area for a defect to be detected"
    )

    max_area = st.sidebar.slider(
        "Maximum Defect Area (px²)",
        min_value=1000.0,
        max_value=100000.0,
        value=50000.0,
        step=1000.0,
        help="Maximum area for a defect to be detected"
    )

    viz_mode = st.sidebar.selectbox(
        "Visualization Mode",
        options=['both', 'bbox', 'contour', 'heatmap'],
        index=0,
        help="How to visualize detected defects"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This system uses advanced computer vision techniques to detect defects in microscope images.

    **Detection Methods:**
    - Adaptive thresholding
    - Edge detection
    - Morphological analysis
    - Texture analysis

    **Supported Formats:**
    - JPG, PNG, TIFF, BMP
    """)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Detect", "📊 Batch Processing", "ℹ️ Help"])

    with tab1:
        st.header("Single Image Analysis")

        uploaded_file = st.file_uploader(
            "Choose a microscope image",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp'],
            help="Upload a microscope image for defect detection"
        )

        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_container_width=True)

            # Process button
            if st.button("🔍 Detect Defects", type="primary", use_container_width=True):
                with st.spinner("Processing image... This may take a moment."):
                    # Load processor
                    processor = load_processor(use_ensemble, min_area, max_area, viz_mode)

                    # Process image
                    result, annotated_img = process_uploaded_file(uploaded_file, processor)

                with col2:
                    st.subheader("Detected Defects")
                    st.image(annotated_img, use_container_width=True)

                # Display statistics
                st.markdown("---")
                st.subheader("📈 Detection Results")

                metric_cols = st.columns(4)

                with metric_cols[0]:
                    st.metric("Total Defects", result['defect_count'])

                # Count defects by type
                defect_counts = {}
                for defect in result['defects']:
                    defect_counts[defect.defect_type] = defect_counts.get(defect.defect_type, 0) + 1

                # Show top defect types
                if defect_counts:
                    sorted_types = sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)

                    for i, (defect_type, count) in enumerate(sorted_types[:3], 1):
                        if i < len(metric_cols):
                            with metric_cols[i]:
                                st.metric(defect_type.replace('_', ' ').title(), count)

                # Detailed breakdown
                col_breakdown, col_quality = st.columns([2, 1])

                with col_breakdown:
                    st.subheader("Defect Breakdown")
                    if defect_counts:
                        chart_data = pd.DataFrame({
                            'Defect Type': list(defect_counts.keys()),
                            'Count': list(defect_counts.values())
                        })
                        st.bar_chart(chart_data.set_index('Defect Type'))
                    else:
                        st.info("No defects detected")

                with col_quality:
                    st.subheader("Image Quality")
                    metadata = result['metadata']
                    st.metric("Focus Quality", f"{metadata.get('focus_quality', 0):.2%}")
                    st.metric("Contrast", f"{metadata.get('contrast', 0):.2%}")
                    st.metric("Noise Level", f"{metadata.get('noise_level', 0):.2%}")

                # Detailed defect table
                st.markdown("---")
                st.subheader("📋 Detailed Defect List")
                display_defect_table(result['defects'])

                # Download button for annotated image
                st.markdown("---")
                annotated_pil = Image.fromarray(annotated_img)
                buf = io.BytesIO()
                annotated_pil.save(buf, format='PNG')
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=buf.getvalue(),
                    file_name=f"annotated_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True
                )

    with tab2:
        st.header("Batch Processing")
        st.info("Upload multiple images (coming soon) or use the command-line interface for batch processing.")

        st.markdown("""
        ### Using the CLI for Batch Processing

        Process multiple images at once using the command line:

        ```bash
        python detect_defects.py --input images/ --output results/
        ```

        This will:
        - Process all images in the `images/` directory
        - Save annotated images to `results/annotated/`
        - Generate CSV and JSON reports
        - Create a summary text file
        """)

    with tab3:
        st.header("Help & Documentation")

        st.markdown("""
        ### How to Use

        1. **Configure Settings** (Optional)
           - Adjust detection parameters in the sidebar
           - Use ensemble detection for best results

        2. **Upload Image**
           - Click "Browse files" or drag and drop
           - Supported formats: JPG, PNG, TIFF, BMP

        3. **Detect Defects**
           - Click "Detect Defects" button
           - Wait for processing to complete
           - Review results and download annotated image

        ### Detection Parameters

        **Minimum Defect Area**: Smaller values detect more defects but may include noise

        **Maximum Defect Area**: Prevents detecting very large regions as single defects

        **Ensemble Detection**: Uses multiple algorithms for better accuracy (recommended)

        ### Defect Types

        - **Scratch**: Linear defects with high aspect ratio
        - **Void**: Dark regions indicating material absence
        - **Particle**: Bright spots indicating foreign material
        - **Edge Defect**: Irregular edges or boundaries
        - **Texture Anomaly**: Surface roughness variations
        - **Threshold Anomaly**: General brightness anomalies

        ### Tips for Best Results

        - Use high-resolution images
        - Ensure consistent lighting
        - Keep images in focus
        - Avoid motion blur

        ### Troubleshooting

        **Too many false positives?**
        - Increase minimum defect area
        - Check image quality metrics

        **Missing defects?**
        - Decrease minimum defect area
        - Enable ensemble detection
        - Check if defect size is within area range

        ### Command-Line Interface

        For batch processing and automation, use the CLI:

        ```bash
        # Process single image
        python detect_defects.py -i image.jpg -o results/

        # Process directory
        python detect_defects.py -i images/ -o results/

        # With custom settings
        python detect_defects.py -i images/ -o results/ --min-area 20 --ensemble
        ```

        For more information, see `USAGE.md` in the project repository.
        """)


if __name__ == '__main__':
    main()
