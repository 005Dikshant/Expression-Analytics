from PIL import Image
import os

# Define the path to the dataset folder
dataset_path = "/path/to/your/dataset/folder"

# Define the folders containing images
image_folders = ["angry", "boredom", "neutral", "engaged"]

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
            # Convert the image to grayscale
            img_grayscale = img.convert("L")

            # Save the grayscale image, overwriting the original
            img_grayscale.save(image_path)

print("Images converted to grayscale successfully.")
