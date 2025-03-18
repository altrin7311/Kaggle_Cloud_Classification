from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalize pixel values

# Training Generator
train_generator = train_datagen.flow_from_directory(
    "/Users/altrintitus/Desktop/understanding_cloud_organization/train_images/",
    target_size=(128, 128),
    batch_size=64,
    class_mode="sparse",
    subset="training"
)

# Validation Generator
val_generator = train_datagen.flow_from_directory(
    "/Users/altrintitus/Desktop/understanding_cloud_organization/train_images/",
    target_size=(128, 128),
    batch_size=64,
    class_mode="sparse",
    subset="validation"
)