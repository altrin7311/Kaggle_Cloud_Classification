import os

image_folder = "/Users/altrintitus/Desktop/understanding_cloud_organization/train_images/"
image_files = os.listdir(image_folder)

# Print available images
print("Available images:", image_files)

# Select the first image dynamically
sample_image_path = os.path.join(image_folder, image_files[0])
print("Using image:", sample_image_path)

import cv2
import matplotlib.pyplot as plt

# Function to load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image could not be loaded")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    return img

# Load and display the image
image = preprocess_image(sample_image_path)
if image is not None:
    plt.imshow(image)
    plt.title("Preprocessed Image")
    plt.axis("off")
    plt.show()