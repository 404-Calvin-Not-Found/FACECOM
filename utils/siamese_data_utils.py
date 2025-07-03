import tensorflow as tf
import numpy as np
import os
import random
from pathlib import Path
from config import ModelConfig, PathConfig
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ================= GPU OPTIMIZATION =================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ================= UTILS =================
def verify_directory(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} directory not found at: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} path is not a directory: {path}")
    return path

@tf.function
def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, ModelConfig.IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return resnet_preprocess(img)

# ================= DATA STRUCTURE LOADING =================
def load_task_b_structure(directory):
    identity_folders = [d for d in directory.iterdir() if d.is_dir()]
    data = defaultdict(list)

    for identity_folder in tqdm(identity_folders, desc="Scanning identities"):
        distortion_folder = identity_folder / "distortion"
        distortions_map = defaultdict(list)

        # Load clean images
        for img_path in identity_folder.glob("*.*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if img_path.parent.name == "distortion":
                continue
            img_id = img_path.stem
            data[identity_folder.name].append({
                "clean": str(img_path),
                "img_id": img_id,
                "distortions": []
            })

        # Load distortions and match with corresponding clean images
        if distortion_folder.exists():
            for dist_img in distortion_folder.glob("*.*"):
                if dist_img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                ref_id = dist_img.stem.rsplit("_", 1)[0]  # match to img_id
                for entry in data[identity_folder.name]:
                    if entry["img_id"] == ref_id:
                        entry["distortions"].append(str(dist_img))
                        break

    return data

# ================= PAIR GENERATION =================
def generate_pairs(task_b_data):
    pos_pairs = []
    neg_pairs = []
    identities = list(task_b_data.keys())

    for identity, entries in task_b_data.items():
        for entry in entries:
            clean_img = entry["clean"]
            for dist_img in entry["distortions"]:
                pos_pairs.append((clean_img, dist_img))

    while len(neg_pairs) < len(pos_pairs):
        id1, id2 = random.sample(identities, 2)
        if not task_b_data[id1] or not task_b_data[id2]:
            continue
        clean_entry = random.choice(task_b_data[id1])
        dist_entry = random.choice(task_b_data[id2])
        if dist_entry["distortions"]:
            neg_pairs.append((clean_entry["clean"], random.choice(dist_entry["distortions"])))

    pairs = pos_pairs + neg_pairs
    labels = [1]*len(pos_pairs) + [0]*len(neg_pairs)
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    return [p for p, _ in combined], [l for _, l in combined]

# ================= DATA LOADER =================
def create_dataset(pairs, labels):
    pair1 = tf.constant([p[0] for p in pairs])
    pair2 = tf.constant([p[1] for p in pairs])
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices(((pair1, pair2), labels))
    dataset = dataset.map(
        lambda paths, label: (
            (preprocess_image(paths[0]), preprocess_image(paths[1])),
            label
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    @tf.function
    def augment(images):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, 0.1)
        return images

    dataset = dataset.map(
        lambda x, y: ((augment(x[0]), augment(x[1])), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(ModelConfig.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ================= MAIN ENTRY =================
def load_siamese_data():
    print("\nLoading Task B image data...")
    train_data = load_task_b_structure(PathConfig.TASK_B_TRAIN)
    val_data = load_task_b_structure(PathConfig.TASK_B_VAL)

    print("\nGenerating pairs...")
    train_pairs, train_labels = generate_pairs(train_data)
    val_pairs, val_labels = generate_pairs(val_data)

    print(f"\nDataset Stats:")
    print(f"Train: {len(train_pairs)} pairs | Pos: {sum(train_labels)} | Neg: {len(train_labels)-sum(train_labels)}")
    print(f"Val:   {len(val_pairs)} pairs | Pos: {sum(val_labels)} | Neg: {len(val_labels)-sum(val_labels)}")

    train_ds = create_dataset(train_pairs, train_labels)
    val_ds = create_dataset(val_pairs, val_labels)

    return train_ds, val_ds
