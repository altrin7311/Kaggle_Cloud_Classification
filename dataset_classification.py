'''
To segregate the images in this format
train_images/
│── Fish/
│   ├── 0011165.jpg
│   ├── 002be4f.jpg
│── Flower/
│   ├── 0011165.jpg
│   ├── 00498ec.jpg
│── Gravel/
│   ├── 0011165.jpg
│   ├── 0031ae9.jpg
│── Sugar/
│   ├── 002be4f.jpg
│   ├── 0035239.jpg
'''

import os
import pandas as pd
import shutil

# Define paths
DATA_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/"
CSV_PATH = os.path.join(DATA_PATH, "processed_data.csv")
IMAGE_FOLDER = os.path.join(DATA_PATH, "train_images/")

# Read CSV file
df = pd.read_csv(CSV_PATH)

# Extract unique classes from "Image_Label"
df["Class"] = df["Image_Label"].apply(lambda x: x.split("_")[-1])  # Extract last part (e.g., "Fish", "Flower")

# Create class folders
unique_classes = df["Class"].unique()
for class_name in unique_classes:
    os.makedirs(os.path.join(IMAGE_FOLDER, class_name), exist_ok=True)

# Move images to respective class folders
for _, row in df.iterrows():
    image_name = row["Image_Label"].split("_")[0]  # Extract base image name (e.g., "0011165.jpg")
    class_name = row["Class"]
    
    src = os.path.join(IMAGE_FOLDER, image_name)  # Source path
    dest = os.path.join(IMAGE_FOLDER, class_name, image_name)  # Destination path
    
    if os.path.exists(src):  # Only move if the file exists
        shutil.move(src, dest)
        print(f"Moved {image_name} → {class_name}/")

print("✅ Images sorted into class folders!")

import os

train_path = "/Users/altrintitus/Desktop/understanding_cloud_organization/train_images/"
print("✅ Classes inside train_images/:", os.listdir(train_path))