import os

# Define the path to your folder
folder_path = 'neutral'

# List all files in the folder
files = os.listdir(folder_path)

# Filter out image files (you can add more extensions if needed)
image_extensions = ['.jpg', '.png', '.jpeg', '.gif']
filtered_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]

# Sort the filtered files to ensure consistency in renaming
filtered_files.sort()

# Rename the images to a1, a2, ..., an
for i, file in enumerate(filtered_files, start=1):
    new_name = f'n{i}{os.path.splitext(file)[1]}'  # Construct new name
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f'Renamed: {old_path} -> {new_path}')