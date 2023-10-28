# Emotion Recognition Project

This project focuses on emotion recognition using a mixed dataset of images featuring angry, boredom, neutral, and engaged expressions. The dataset is organized within the `new_dataset` folder, further divided into respective emotion categories.

## Contents

- `new_dataset/`
  - `angry/`
  - `boredom/`
  - `neutral/`
  - `engaged/`
  - `rename.py`
- `rescale_images.py`
- `convert_into_grayscale.py`
- `image_sharpening.py`
- `data_visualization.py`
- `README.md`
- `Report.pdf`
- `Originality Form.pdf`

## Instructions

### Data Cleaning:

1. **Rescale Images:**
   - Use `rescale_images.py` to standardize the dimensions of all images to 256x256 pixels.

2. **Convert to Grayscale:**
- Apply grayscale conversion using `convert_into_grayscale.py` to simplify color information.

3. **Noise Reduction:**
- Use `image_sharpening.py` to reduce noise and enhance image clarity.

4. **Rename Images:**
- Execute `rename.py` located in the `new_dataset` folder to standardize image names with emotion initials and numerical identifiers.

### Data Visualization:

1. **Class Distribution:**
- Run `data_visualization.py` to visualize the distribution of images in each class.
  
2. **Sample Images:**
- The script will display a 5x5 grid with randomly chosen images from each class.

3. **Pixel Intensity Distribution:**
- The script will generate histograms showcasing pixel intensity distribution, including RGB channels for color images.

## Note

Please ensure that you have Python and the required libraries installed to execute the scripts. Additionally, verify that the file paths and directories are set up correctly before running the code.

