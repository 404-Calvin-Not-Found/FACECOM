import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import PathConfig, ModelConfig
from pathlib import Path

# ================= DIRECTORY CHECK =================
def verify_directory(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} directory not found at: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} path is not a directory: {path}")
    return path

# ================= GENDER CLASSIFICATION DATA =================
def load_gender_classification_data():
    """Load training and validation data generators for gender classification."""
    train_dir = verify_directory(PathConfig.TASK_A_TRAIN, "Training")
    val_dir = verify_directory(PathConfig.TASK_A_VAL, "Validation")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=ModelConfig.IMAGE_SIZE,
        batch_size=ModelConfig.BATCH_SIZE * 2,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=ModelConfig.IMAGE_SIZE,
        batch_size=ModelConfig.BATCH_SIZE * 2,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen

if __name__ == "__main__":
    print("\nTesting gender classification data loading...")
    try:
        train_gen, val_gen = load_gender_classification_data()
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
