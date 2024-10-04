# Computer Vision Scripts

This repository contains Python scripts demonstrating various computer vision techniques using OpenCV, NumPy, and Matplotlib.

## Requirements

- Python 3.9
- OpenCV 4 (at least version 3)
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Computer-Vision-Scripts.git
   cd Computer-Vision-Scripts
   ```
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```
   pip install opencv-python numpy matplotlib
   ```

## Usage

Run each script independently:

```
python script_name.py
```

## Notable Experiments

### 1. Color Histogram Computation from ROI

Computes color histograms from a Region of Interest in a webcam feed.

**Key Features:**
- Real-time processing with user-selectable ROI
- Multiple color space support (BGR, HSV, YCrCb)

### 2. Color-based Object Tracking

Tracks objects based on color using dynamic thresholding.

**Key Features:**
- Real-time tracking with color-based segmentation
- Morphological filtering for improved detection

### 3. Invisibility Cloak

Creates an "invisibility cloak" effect inspired by Harry Potter.

**Key Features:**
- Background subtraction and color-based segmentation
- Real-time invisibility effect

### 4. Histogram Equalization for BGR Images

Enhances contrast in color images using histogram equalization.

**Key Features:**
- Supports BGR color images
- Before and after comparison visualization

### 5. Vertical Pattern Segmentation

Segments vertical patterns in images using advanced filtering techniques.

**Key Features:**
- Combines Gabor filters, Otsu thresholding, and morphological operations
- Specialized for vertical pattern extraction