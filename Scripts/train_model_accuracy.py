"""
This script:
1. Loads and preprocesses test images from the `test_images/` folder.
2. Loads the true labels from `test.csv`.
3. Uses the trained model to make predictions.
4. Computes accuracy by comparing predictions with actual labels.
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array

# === Define Paths ===
DATA_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/"
IMAGE_FOLDER = os.path.join(DATA_PATH, "test_images/")  # Test images folder
MODEL_PATH = os.path.join(DATA_PATH, "cloud_classification_model.h5")  # Trained Model
CSV_PATH = os.path.join(DATA_PATH, "test.csv")  # True Labels

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("\n✅ Model Loaded Successfully!")

# Image dimensions (same as training)
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 256  # Process images in batches

# Function to load and preprocess a single image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    if img is None:
        return None  # Skip if missing
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize
    img = img_to_array(img) / 255.0  # Normalize pixel values
    return img

# Load true labels from CSV
df_test = pd.read_csv(CSV_PATH)  # Ensure this file exists
image_label_dict = dict(zip(df_test["Image_Label"], df_test["EncodedPixels"]))  # Adjust as needed

# Get all test image file paths
image_files = [os.path.join(IMAGE_FOLDER, img) for img in os.listdir(IMAGE_FOLDER) if img.endswith((".jpg", ".png"))]

# Process images and compute accuracy
correct_predictions = 0
total_images = 0

for i in range(0, len(image_files), BATCH_SIZE):
    batch_files = image_files[i:i+BATCH_SIZE]
    batch_images = np.array([preprocess_image(img) for img in batch_files if preprocess_image(img) is not None])

    if batch_images.size > 0:  # Ensure images exist in batch
        batch_predictions = model.predict(batch_images)
        batch_labels = np.argmax(batch_predictions, axis=1)

        for img, predicted_label in zip(batch_files, batch_labels):
            img_name = os.path.basename(img)
            true_label = image_label_dict.get(img_name, None)

            if true_label is not None:
                total_images += 1
                if int(true_label) == int(predicted_label):
                    correct_predictions += 1

# Compute accuracy
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0

print(f"\n✅ Model Accuracy: {accuracy:.2f}%")
print("\n✅ Model Evaluation on Test Images Completed! 🚀")