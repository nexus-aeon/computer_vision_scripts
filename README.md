# Computer Vision Scripts

This repository contains a collection of Python scripts exploring various computer vision projects and algorithms. These scripts are designed to demonstrate practical applications of computer vision techniques using OpenCV, NumPy, and Matplotlib.

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

Each script in this repository is self-contained and can be run independently. To execute a script, use the following command:

```
python script_name.py
```

Replace `script_name.py` with the name of the script you want to run.

## Notable Experiments

### 1. Color Histogram Computation from ROI

This script computes color histograms from a Region of Interest (ROI) in a webcam feed. It supports various color spaces including BGR, HSV, and YCrCb.

**Features:**
- Real-time webcam feed processing
- User-selectable ROI
- Support for multiple color spaces
- Histogram visualization

### 2. Color-based Object Tracking

This script performs object tracking based on color. It uses the color histogram script to determine color thresholds for the target object.

**Features:**
- Dynamic color thresholding
- Image segmentation
- Morphological filtering (Closing and Dilation)
- Real-time object tracking visualization

### 3. Invisibility Cloak

Inspired by Harry Potter, this script creates an "invisibility cloak" effect. It uses a uniformly colored cloth to render the user invisible.

**Features:**
- Background capture and subtraction
- Color-based segmentation
- Real-time video processing
- Magical invisibility effect

### 4. Histogram Equalization for BGR Images

This script demonstrates histogram equalization on BGR (color) images to enhance contrast and improve image quality.

**Features:**
- Support for BGR color images
- Histogram computation and equalization
- Before and after comparison visualization