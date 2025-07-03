import os
from pathlib import Path

# ================= GPU CONFIGURATION =================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# ================= BASE PATHS =================
BASE_DIR = Path("D:/FACECOM")
DATASET_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ================= MODEL PARAMETERS =================
class ModelConfig:
    # Image processing
    IMAGE_SIZE = (224, 224)
    CHANNELS = 3

    # Training parameters
    BATCH_SIZE = 64
    EMBEDDING_SIZE = 256
    INITIAL_LR = 0.0001
    WEIGHT_DECAY = 0.0001
    EPOCHS = 50

    # Augmentation
    ROTATION_RANGE = 15
    BRIGHTNESS_RANGE = 0.1

    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.001

# ================= COMPLETE PATH CONFIG =================
class PathConfig:
    # Root directories
    ROOT = BASE_DIR
    MODELS = MODELS_DIR
    RESULTS = RESULTS_DIR

    # Dataset paths
    DATASETS = DATASET_DIR
    TASK_A_TRAIN = DATASET_DIR / "Task_A" / "train"
    TASK_A_VAL = DATASET_DIR / "Task_A" / "val"
    TASK_B_TRAIN = DATASET_DIR / "Task_B" / "train"
    TASK_B_VAL = DATASET_DIR / "Task_B" / "val"

    # Model paths
    GENDER_MODEL = MODELS_DIR / "gender_classifier.h5"
    SIAMESE_MODEL = MODELS_DIR / "siamese_model.keras"
    EMBEDDING_MODEL = MODELS_DIR / "embedding_model.keras"
    FACE_MODEL = MODELS_DIR / "resnet_face_model.keras"  # <-- Updated correctly

    # Results subdirectories
    TRAINING_LOGS = RESULTS_DIR / "training_logs"
    EMBEDDING_VIS = RESULTS_DIR / "embeddings"
    GENDER_RESULTS = RESULTS_DIR / "gender"
    FACE_RESULTS = RESULTS_DIR / "face"
    TRAINING_CURVES = RESULTS_DIR / "training_curves.jpg"

def initialize_directories():
    """Create all required directories"""
    directories = [
        MODELS_DIR,
        RESULTS_DIR,
        PathConfig.TRAINING_LOGS,
        PathConfig.EMBEDDING_VIS,
        PathConfig.GENDER_RESULTS,
        PathConfig.FACE_RESULTS
    ]

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {dir_path}")

# Initialize directories when config is imported
initialize_directories()

if __name__ == "__main__":
    print("Configuration Verification:")
    print(f"Using TensorFlow with GPU acceleration: {os.getenv('TF_GPU_ALLOCATOR', 'Not set')}")

    # Verify paths
    required_paths = [
        PathConfig.TASK_A_TRAIN,
        PathConfig.TASK_A_VAL,
        PathConfig.TASK_B_TRAIN,
        PathConfig.TASK_B_VAL,
        PathConfig.GENDER_RESULTS,
        PathConfig.FACE_RESULTS
    ]

    for path in required_paths:
        exists = path.exists()
        print(f"{'✅' if exists else '❌'} {path.name}: {'Found' if exists else 'Missing'}")
