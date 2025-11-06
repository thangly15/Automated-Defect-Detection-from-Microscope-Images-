# Quick Start Guide for Beginners

This is the simplest way to get started! Follow these 3 easy steps.

## Step 1: Install Requirements (One Time Only)

Open your terminal and run:

```bash
pip install opencv-python numpy pandas pyyaml tqdm
```

**That's it!** You now have everything you need.

---

## Step 2: Run the Simple Script

### Option A: Interactive Mode (Easiest!)

Just run the script and it will ask you questions:

```bash
python simple_detect.py
```

Then:
1. Enter the path to your images folder
2. Choose default or custom settings
3. Wait for it to finish
4. Check the `results_simple/` folder!

### Option B: Quick Mode

If you already know your folder path:

```bash
python simple_detect.py path/to/your/images/
```

---

## Step 3: View Results

Look in the `results_simple/` folder:
- **result_*.jpg** = Your images with defects highlighted
- **summary.txt** = Text report with all defect counts

**Colors in images:**
- 🔴 Red = Scratches
- 🔵 Blue = Voids (holes)
- 🟢 Green = Particles
- 🟠 Orange = Edge problems
- 🟣 Magenta = Texture problems
- 🟡 Cyan = Brightness problems

---

## Example

```bash
# Create a test folder
mkdir my_microscope_images

# Copy your images there
# (copy your .jpg or .png files)

# Run detection
python simple_detect.py my_microscope_images/

# Check results
ls results_simple/
```

---

## What If I Want More Control?

### Method 1: Change Settings in Simple Script

When you run `python simple_detect.py`, choose option 2:
- Change minimum defect size (smaller = more sensitive)
- Turn off ensemble mode (faster but less accurate)

### Method 2: Use Python Code

Create your own script:

```python
import cv2
from src.defect_detector import create_detector

# Create detector
detector = create_detector(
    min_area=15,        # Minimum defect size
    max_area=50000,     # Maximum defect size
    use_ensemble=True   # True = more accurate, False = faster
)

# Load your image
image = cv2.imread("my_image.jpg")

# Find defects
defects = detector.detect_defects(image)

# Print results
print(f"Found {len(defects)} defects!")

for i, defect in enumerate(defects, 1):
    print(f"  {i}. Type: {defect.defect_type}, Size: {defect.area:.0f} pixels")
```

### Method 3: Try the Advanced CLI

For more features:

```bash
python detect_defects.py --input images/ --output results/ --min-area 20
```

See all options:
```bash
python detect_defects.py --help
```

---

## Troubleshooting

### "No module named cv2"

Install OpenCV:
```bash
pip install opencv-python
```

### "Folder not found"

Make sure you type the full path. Examples:
- Linux/Mac: `/home/user/images/` or `./images/`
- Windows: `C:\Users\YourName\Desktop\images\` or `.\images\`

### Too many false positives (noise detected as defects)

Increase minimum size:
```bash
python simple_detect.py
# Choose option 2
# Enter bigger number like 20 or 30
```

### Missing small defects

Decrease minimum size:
```bash
python simple_detect.py
# Choose option 2
# Enter smaller number like 5
```

---

## Next Steps

Once you're comfortable with the simple script:

1. **Try the Web Interface** (visual and interactive):
   ```bash
   pip install streamlit
   streamlit run streamlit_app.py
   ```

2. **Read the Full Documentation**: See `USAGE.md` for all features

3. **Customize Detection**: Edit the detector settings in `src/defect_detector.py`

---

## Files Explained (For Beginners)

| File | What It Does | When To Use |
|------|--------------|-------------|
| `simple_detect.py` | Easiest script | Start here! |
| `detect_defects.py` | Advanced CLI | When you need more options |
| `streamlit_app.py` | Web interface | Visual, interactive use |
| `demo.py` | Test without real images | Testing the system |
| `src/defect_detector.py` | Main detection code | Want to understand how it works |

---

## Summary - Three Ways to Use This

### 1. Super Simple (Recommended for Beginners)
```bash
python simple_detect.py
```

### 2. Command Line (For More Control)
```bash
python detect_defects.py -i images/ -o results/
```

### 3. Web Interface (Visual & Interactive)
```bash
streamlit run streamlit_app.py
```

**Start with #1, then try #3, then move to #2 when you need automation!**
