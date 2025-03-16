# Kaggle_Cloud_Classification

EfficientNet-based model for classifying cloud formations from satellite images, following Kaggle’s “Understanding Clouds” competition format.

---

##  Overview
This repository contains:
- A **fine-tuning script** for **EfficientNetB0** with a custom classifier head.
- **Data loaders** using `ImageDataGenerator` for efficient batch processing and augmentation.
- A **submission script** to create a valid `submission.csv` file for Kaggle.
- Class index mappings saved during training for consistent label handling.

---

##  Requirements
Ensure you have the following dependencies installed:
- **Python 3.8+**
- **TensorFlow 2.x**
- **OpenCV** (`pip install opencv-python`)
- **scikit-learn**
- **pandas, numpy, matplotlib**

---

## Setup & Usage

### Step 1: Install dependencies
```sh
pip install -r requirements.txt
