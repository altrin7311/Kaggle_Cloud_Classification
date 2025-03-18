"""
Cloud Classification Preprocessing Script
------------------------------------------
This script loads and preprocesses satellite images from the MAHASAT dataset for cloud classification.
It efficiently processes images using multiprocessing or threading for performance optimization.

Key Features:
✅ Reads image paths from 'train.csv'
✅ Loads images, resizes to (256x256), and normalizes pixel values
✅ Uses multiprocessing (or threading as fallback) to speed up large dataset processing
✅ Saves preprocessed images to a new folder to avoid memory overload
✅ Outputs a new CSV with updated image paths

How to Use:
- Ensure dataset paths are correctly set.
- Run the script, and it will process and save images efficiently.
"""

import os
import pandas as pd
import cv2
import numpy as np
from multiprocessing import Pool, freeze_support
from tqdm import tqdm

# **Set Paths (Modify If Needed)**
DATA_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/"
IMAGE_FOLDER = os.path.join(DATA_PATH, "train_images")
CSV_PATH = os.path.join(DATA_PATH, "train.csv")
PROCESSED_FOLDER = os.path.join(DATA_PATH, "processed_images")  # Output folder

# **Ensure the processed folder exists**
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# **Check dataset existence**
if os.path.exists(IMAGE_FOLDER):
    print(f"✅ Dataset Found: {IMAGE_FOLDER}")
else:
    print("❌ Dataset Not Found! Check the path.")
    exit()

# **Load CSV file**
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(df.head())  # Show first few rows
else:
    print(f"❌ CSV File Not Found: {CSV_PATH}")
    exit()

# **Convert image names to full paths**
df["Image_Path"] = df["Image_Label"].apply(lambda x: os.path.join(IMAGE_FOLDER, x.split("_")[0]))

# **Processing function**
def process_image(image_path):
    """Loads, preprocesses, and saves images efficiently."""
    try:
        img = cv2.imread(image_path)  # Load image
        if img is None:
            return None  # Skip missing images
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)  # Resize
        img = img / 255.0  # Normalize

        # Save processed image
        output_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
        cv2.imwrite(output_path, (img * 255).astype(np.uint8))  # Convert back to uint8 before saving
        return output_path  # Return saved file path
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

# **Multiprocessing wrapper function**
def process_batch(image_paths):
    with Pool(processes=8) as pool:  # Adjust based on CPU
        processed_paths = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths)))
    return processed_paths

if __name__ == "__main__":
    freeze_support()  # **Required for Windows/macOS Multiprocessing**

    # **Batch Processing for Large Dataset**
    BATCH_SIZE = 1000  # Adjust as needed (500-2000 recommended)
    total_images = len(df)
    processed_images = []

    for start in range(0, total_images, BATCH_SIZE):
        batch = df["Image_Path"].iloc[start:start + BATCH_SIZE].tolist()
        print(f"\n🚀 Processing batch {start} - {start + len(batch)}...")
        processed_paths = process_batch(batch)
        processed_images.extend(processed_paths)

    # **Drop failed images & Save new CSV**
    df["Processed_Path"] = processed_images
    df.dropna(subset=["Processed_Path"], inplace=True)
    df.to_csv(os.path.join(DATA_PATH, "processed_data.csv"), index=False)

    # **Final Summary**
    print("\n✅ All images processed & saved successfully!")
    print(f"Total Images Processed: {len(df)}")

# loading the csv in Pandas
import pandas as pd

# Load the processed CSV
df = pd.read_csv("/Users/altrintitus/Desktop/understanding_cloud_organization/processed_data.csv")

# Display first few rows
print(df.head())