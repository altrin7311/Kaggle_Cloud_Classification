"""
Kaggle Submission Script for 'Understanding Clouds from Satellite Images'

This script:
1. Loads the trained EfficientNetB0 model.
2. Preprocesses test images in batches.
3. Performs inference using the trained model.
4. Formats predictions in the Kaggle submission format.
5. Saves the results as a CSV file (`submission.csv`).

Expected Output:
- A CSV file (`submission.csv`) with:
  - "Image_Label": The original image name with the predicted class.
  - "EncodedPixels": The run-length encoded (RLE) segmentation mask.

Note: If no cloud is detected, use "1 1" as per Kaggle rules.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from fine_tuning_Generator_ver3 import train_generator

class_indices = train_generator.class_indices  # Get class mappings
index_to_class = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Paths
DATA_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/"
TEST_PATH = os.path.join(DATA_PATH, "test_images/")
MODEL_PATH = os.path.join(DATA_PATH, "fine_tuned_cloud_model_30epochs.h5")
SUBMISSION_FILE = os.path.join(DATA_PATH, "submission.csv")

import json

# Load class indices from saved JSON file
class_indices_path = os.path.join(DATA_PATH, "class_indices.json")
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary: Map index → class name
index_to_class = {v: k for k, v in class_indices.items()}

# Image Parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 64  # Process images in batches

# Load the trained model
print("✅ Loading Trained Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")

# Load test images
test_images = sorted(os.listdir(TEST_PATH))
test_image_paths = [os.path.join(TEST_PATH, img) for img in test_images]

# Define Image Preprocessing
def preprocess_image(image_path):
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Warning: Could not read {image_path}")
        return None
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize
    return img

# Batch processing
predictions = []
batch_images = []
batch_filenames = []

print("✅ Processing Test Images in Batches...")
for i, image_path in enumerate(test_image_paths):
    img = preprocess_image(image_path)
    if img is not None:
        batch_images.append(img)
        batch_filenames.append(image_path)

    # Process batch when full
    if len(batch_images) == BATCH_SIZE or i == len(test_image_paths) - 1:
        batch_images_np = np.array(batch_images)
        batch_preds = model.predict(batch_images_np)
        batch_labels = np.argmax(batch_preds, axis=1)  # Get class labels

        # Store results
        for img_path, label in zip(batch_filenames, batch_labels):
            image_name = os.path.basename(img_path)
            # Retrieve class names from train_generator (NOT from model)
            class_indices = train_generator.class_indices
            class_name = list(class_indices.keys())[label]  # Get class name
            image_label = f"{image_name}_{class_name}"  # Append predicted class

            # No segmentation mask available, so we use "1 1" for no cloud detected
            encoded_pixels = "1 1"

            predictions.append([image_label, encoded_pixels])

        # Clear batch
        batch_images, batch_filenames = [], []

# Convert to DataFrame
submission_df = pd.DataFrame(predictions, columns=["Image_Label", "EncodedPixels"])

# Save CSV file
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"✅ Submission File Saved: {SUBMISSION_FILE}")