from PIL import Image, ImageFilter
import os

# Define the path to the dataset folder
dataset_path = "new_dataset"

# Define the folders containing images
image_folders = ["angry", "boredom", "neutral", "engaged"]

# Define a sharpening filter
sharpening_filter = ImageFilter.SHARPEN

# Iterate through each image folder
for folder in image_folders:
    folder_path = os.path.join(dataset_path, folder)

    # Get list of images in the folder
    images = os.listdir(folder_path)

    # Iterate through each image
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)

        # Open the image using Pillow
        with Image.open(image_path) as img:
            # Apply the sharpening filter
            img_sharpened = img.filter(sharpening_filter)

            # Save the sharpened image, overwriting the original
            img_sharpened.save(image_path)

print("Images sharpened successfully.")
