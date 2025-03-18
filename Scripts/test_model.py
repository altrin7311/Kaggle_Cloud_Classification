import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Define paths
DATA_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/"
MODEL_PATH = os.path.join(DATA_PATH, "fine_tuned_cloud_model_30epochs.h5")
TEST_IMAGE_PATH = "/Users/altrintitus/Desktop/understanding_cloud_organization/test_images/0a16aa9.jpg"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("\n✅ Model Loaded Successfully!")

# Preprocess test image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found")
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand for batch processing
    return img

# Load and predict
image = preprocess_image(TEST_IMAGE_PATH)
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# Define class labels
class_labels = ['Sugar', 'Flower', 'Gravel', 'Fish']
predicted_label = class_labels[predicted_class]

# Display result
plt.imshow(cv2.imread(TEST_IMAGE_PATH)[:, :, ::-1])
plt.title(f"Predicted Class: {predicted_label}")
plt.axis("off")
plt.show()
print(f"\n✅ Prediction: {predicted_label}")