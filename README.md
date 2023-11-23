# Alignment Project

This repository contains a Python script for aligning scanned documents using keypoint matching and homography estimation. The code is designed to work with two images: a scanned document (`scanned_document.png`) and an original document (`original_document.png`). The alignment process involves detecting keypoints, matching descriptors, applying a ratio test to filter matches, estimating a homography matrix, and warping the perspective to align the images.

## Dependencies

Make sure you have the necessary Python libraries installed:

    ```bash
    pip install opencv-python numpy matplotlib
    ```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/alignment-project.git
    cd alignment-project

2. Upload the scanned document (scanned_document.png) and the original document (original_document.png) to the project folder.

3. Run the script:
    ```bash
    python image-alignment.py
    ```

- The script will display visualizations of the scanned and original documents side by side before and after alignment.