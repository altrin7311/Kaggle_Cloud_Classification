"""
Fine-Tuning EfficientNetB0 for Cloud Classification (30 Epochs - Classified Data)

This script fine-tunes the EfficientNetB0 model using images categorized into class folders.
It uses an ImageDataGenerator for dynamic image loading and augmentation, improving training efficiency.
The model is initially loaded with pre-trained weights and fine-tuned with a custom classifier head.

Key Features:
- Uses **ImageDataGenerator** for efficient data loading with augmentation.
- **Splits data dynamically** into 80% training and 20% validation.
- **EfficientNetB0 as base model** with its pre-trained weights.
- **Custom classifier head** with dropout to prevent overfitting.
- **Sparse categorical crossentropy loss** since labels are integers.
- **Early stopping** to prevent overfitting by monitoring validation loss.
- **Fine-tunes for 30 epochs** with a batch size of 64.
- **Saves the fine-tuned model** for future predictions.

Expected Outcome:
- **Accuracy: ~1.0**
- **Loss: Very low (~1.0895e-05)**
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define dataset paths
DATA_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/"
TRAIN_PATH = os.path.join(DATA_PATH, "train_images/")  # Now contains class folders
NEW_MODEL_PATH = os.path.join(DATA_PATH, "fine_tuned_cloud_model_30epochs.h5")

# Image parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 64

# Create ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2  # 80% train, 20% validation
)

# Training Generator (loads from sorted folders)
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="sparse",  # Since labels are integers
    subset="training"
)

# Validation Generator
val_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

# Load Pretrained EfficientNetB0
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
base_model.trainable = False  # Freeze layers initially

# Add custom classifier head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)  # Reduces overfitting
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(len(train_generator.class_indices), activation="softmax")(x)  # Output layer for classification

# Create model
model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model
EPOCHS = 30
print("\n🚀 Fine-Tuning EfficientNetB0 on Cloud Dataset (30 Epochs)...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),  # Ensure full training
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)

# Save the fine-tuned model
model.save(NEW_MODEL_PATH)
print("\n✅ Fine-Tuned Model (30 Epochs) Saved Successfully! 🚀")

import json

# Save class indices to a JSON file
class_indices = train_generator.class_indices
with open(os.path.join(DATA_PATH, "class_indices.json"), "w") as f:
    json.dump(class_indices, f)
print("✅ Class Indices Saved!")

# Accuracy = 1.0, loss = 1.0895e-05