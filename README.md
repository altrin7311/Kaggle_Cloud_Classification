Kaggle_Cloud_Classification

EfficientNet-based model for classifying cloud formations from satellite images, following Kaggle’s “Understanding Clouds” competition format.

Overview

This repository contains:
	•	A fine-tuning script for EfficientNetB0 with a custom classifier head.
	•	Data loaders using ImageDataGenerator for efficient batch processing and augmentation.
	•	A submission script to create a valid CSV file for Kaggle submissions.

Repository Structure

Kaggle_Cloud_Classification/
├── fine_tuning_Generator_ver3.py   (Main training script, saves class_indices.json)
├── submission_with_batches.py      (Generates submission.csv using batch processing)
├── class_indices.json              (Saved class-to-index mappings from training)
├── train_images/                   (Folder with subfolders for each class)
├── test_images/                    (Folder with test images)
├── README.md                       (This README)
└── …

Requirements
	•	Python 3.8+
	•	TensorFlow 2.x
	•	OpenCV (e.g. pip install opencv-python)
	•	scikit-learn
	•	pandas, numpy, matplotlib

Setup & Usage
	1.	Install dependencies:
pip install -r requirements.txt
	2.	Organize data:
	•	Place training images in train_images/<class_name>/.
	•	Place test images in test_images/.
	3.	Train the model:
	•	Run fine_tuning_Generator_ver3.py to train EfficientNetB0 on your dataset.
	•	This script saves fine_tuned_cloud_model_30epochs.h5 and class_indices.json.
	4.	Generate Submission:
	•	Run submission_with_batches.py to create submission.csv.
	•	Upload submission.csv to Kaggle.

Acknowledgments
	•	Kaggle’s “Understanding Clouds” competition
	•	TensorFlow for deep learning framework
	•	EfficientNet for state-of-the-art performance on image classification

License

This project is licensed under the MIT License.
