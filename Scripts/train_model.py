"""
Cloud Classification Model Training

This script:
1. Loads preprocessed image paths and labels from `processed_data.csv`
2. Processes images in **batches** (to handle large datasets)
3. Encodes labels and splits data into **train/test**
4. Builds a **CNN Model** (Convolutional Neural Network)
5. Saves the trained model

Optimized for large datasets (5.8GB+) with batch processing.
"""

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# Define Paths
DATA_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/"
CSV_PATH = os.path.join(DATA_PATH, "processed_data.csv")
IMAGE_FOLDER = os.path.join(DATA_PATH, "processed_images/")

# Load Processed Data
df = pd.read_csv(CSV_PATH)

# Extract Labels & Image Paths
image_paths = df["Image_Path"].values
labels = df["Image_Label"].values

# Image Dimensions
IMG_WIDTH = 128   # Resize width
IMG_HEIGHT = 128  # Resize height
BATCH_SIZE = 512  # Process images in batches to avoid memory overload

# Label Encoding
unique_labels = list(set(labels))
label_map = {label: i for i, label in enumerate(unique_labels)}
encoded_labels = np.array([label_map[label] for label in labels])

# Function to Load and Process Images in Batches
def batch_load_images(image_paths, batch_size=512):
    """Loads images in batches to prevent memory overload."""
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip missing images
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize
            img = img_to_array(img) / 255.0  # Normalize
            images.append(img)

        yield np.array(images)

# Train-Test Split (80% Training, 20% Validation)
X_train_paths, X_val_paths, y_train, y_val = train_test_split(image_paths, encoded_labels, test_size=0.2, random_state=42)

# CNN Model Definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')  # Multi-class classification
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training Loop with Batch Processing
EPOCHS = 10
print("\n🚀 Training Model with Batch Processing...\n")

for epoch in range(EPOCHS):
    print(f"\n🔄 Epoch {epoch+1}/{EPOCHS}")

    for batch_images in batch_load_images(X_train_paths, BATCH_SIZE):
        model.fit(batch_images, y_train[:len(batch_images)], batch_size=BATCH_SIZE, verbose=1)

    print(f"✅ Epoch {epoch+1} Completed!\n")

# Save Model
model.save(os.path.join(DATA_PATH, "cloud_classification_model.h5"))
print("\n✅ Model Training Complete & Saved Successfully! 🎉")

import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("/Users/altrintitus/Desktop/understanding_cloud_organization/cloud_classification_model.h5")

# Load processed test data
X_test = np.load("/Users/altrintitus/Desktop/understanding_cloud_organization/X_test.npy")  # If saved separately
y_test = np.load("/Users/altrintitus/Desktop/understanding_cloud_organization/y_test.npy")  # If saved separately

# Evaluate model on test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

'''
Very bad accuracy 0.05 approx. and high loss 5.700 approx.
'''