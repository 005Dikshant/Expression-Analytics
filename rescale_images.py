from PIL import Image
import os

# Define the path to the dataset folder
dataset_path = "new_dataset"

# Define the target size for resizing
target_size = (256, 256)

# Define the folders containing images
image_folders = ["angry", "boredom", "neutral", "engaged"]

# Iterate through each image folder
for folder in image_folders:
    # Get the list of images in the folder
    images = os.listdir(os.path.join(dataset_path, folder))

    # Iterate through each image
    for image_name in images:
        # Get the image path
        image_path = os.path.join(dataset_path, folder, image_name)

        # Open and resize the image
        with Image.open(image_path) as img:

            img_resized = img.resize(target_size, Image.ANTIALIAS)

            # Save the resized image back to the same path
            img_resized.save(image_path)

print("Images resized successfully.")
