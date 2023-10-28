import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Define the dataset path (base path)
dataset_path = 'new_dataset'
label_names = ['angry', 'boredom', 'engaged', 'neutral']

# Create a 6x5 grid (including the extra 5 blocks)
fig, axes = plt.subplots(6, 5, figsize=(12, 12))

# Initialize lists to store pixel intensities for each channel
red_channel = []
green_channel = []
blue_channel = []

# Initialize class counts dictionary
class_counts = {}

# Iterate over label names
for i, label in enumerate(label_names):
    folder_path = os.path.join(dataset_path, label)
    image_files = os.listdir(folder_path)
    random.shuffle(image_files)

    # Take the first 5 images for display
    for j in range(5):
        image_path = os.path.join(folder_path, image_files[j])
        image = cv2.imread(image_path,
                           cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        axes[i, j].imshow(image, cmap='gray')
        axes[i, j].set_title(label)
        axes[i, j].axis('off')

    # Select a random image for pixel intensity histograms
    random_image_path = os.path.join(folder_path, random.choice(image_files))
    random_image = cv2.imread(random_image_path, cv2.IMREAD_COLOR)

    # Iterate over each channel (BGR order)
    for k in range(3):
        channel_histogram = cv2.calcHist([random_image], [k], None, [256],
                                         [0, 256])
        if k == 0:
            blue_channel = channel_histogram
        elif k == 1:
            green_channel = channel_histogram
        else:
            red_channel = channel_histogram

    # Get list of images in the folder and count them
    num_images = len(image_files)
    class_counts[label] = num_images

# Add extra 5 blocks
for i in range(5):
    axes[-1, i].axis('off')

# Plot histograms
plt.figure(figsize=(10, 6))
plt.plot(red_channel, color='red', label='Red Channel')
plt.plot(green_channel, color='green', label='Green Channel')
plt.plot(blue_channel, color='blue', label='Blue Channel')

plt.title(f'Pixel Intensity Distributions for {label}')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()

# Plot class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Class Distribution')

plt.tight_layout()
plt.show()
