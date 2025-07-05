import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from PIL import Image

# ============== CONFIGURATION ==============
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
MARGIN = 1.0

# ============== L2 NORMALIZATION LAYER ==============
class L2Normalization(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# ============== LOSS AND METRICS ==============
def contrastive_loss(margin=MARGIN):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        squared_pred = tf.square(y_pred)
        margin_squared = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * squared_pred + (1 - y_true) * margin_squared)
    loss.__name__ = 'contrastive_loss'
    return loss

def contrastive_accuracy(threshold=MARGIN):
    def acc_fn(y_true, y_pred):
        pred = tf.cast(y_pred < threshold, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(pred, y_true), tf.float32))
    acc_fn.__name__ = 'contrastive_accuracy'
    return acc_fn

# ============== CALLBACK TO SAVE BEST MODEL ==============
class SaveBestModelCallback(Callback):
    def __init__(self, model, save_path, target_accuracy=0.97):
        super().__init__()
        self.best_acc = 0.0
        self.model = model
        self.save_path = save_path
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_contrastive_accuracy')
        if val_acc is not None and val_acc > self.best_acc and val_acc < self.target_accuracy:
            self.best_acc = val_acc
            print(f"\nSaving new best model with accuracy {val_acc:.4f}...")
            self.model.save(self.save_path)

# ============== CALLBACK TO PLOT LIVE TRAINING ==============
class LivePlotCallback(Callback):
    def __init__(self, save_path="results/acc_plot.jpg"):
        super().__init__()
        self.save_path = save_path
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.train_loss.append(logs.get("loss", 0))
        self.val_loss.append(logs.get("val_loss", 0))
        self.train_acc.append(logs.get("contrastive_accuracy", 0))
        self.val_acc.append(logs.get("val_contrastive_accuracy", 0))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_loss, label="Train Loss")
        plt.plot(self.epochs, self.val_loss, label="Val Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.train_acc, label="Train Accuracy")
        plt.plot(self.epochs, self.val_acc, label="Val Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()

# ============== MODEL DEFINITIONS ==============
def build_embedding_network():
    base_model = applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    for layer in base_model.layers[:80]:
        layer.trainable = False

    inputs = layers.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128)(x)
    x = L2Normalization()(x)
    return models.Model(inputs, x)

def create_siamese_model(embedding):
    input_a = layers.Input(shape=IMAGE_SIZE + (3,))
    input_b = layers.Input(shape=IMAGE_SIZE + (3,))
    embed_a = embedding(input_a)
    embed_b = embedding(input_b)
    distance = layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1, keepdims=True))([embed_a, embed_b])
    return models.Model(inputs=[input_a, input_b], outputs=distance)

# ============== UNFREEZE FUNCTION ==============
def unfreeze_layers(model, percentage):
    base_model = model.layers[2].layers[1]  # model -> embedding -> base_model
    total = len(base_model.layers)
    to_unfreeze = int(total * percentage)
    for layer in base_model.layers[-to_unfreeze:]:
        layer.trainable = True
    print(f"\nUnfroze last {to_unfreeze}/{total} layers")

# ============== CUSTOM DATA LOADER FOR JPG ==============
def load_task_b_structure(directory):
    identity_folders = [d for d in Path(directory).iterdir() if d.is_dir()]
    data = defaultdict(list)

    for identity_folder in tqdm(identity_folders, desc=f"Scanning {directory} identities"):
        distortion_folder = identity_folder / "distortion"
        distortion_images = list(distortion_folder.glob("*.jpg")) if distortion_folder.exists() else []
        distortion_map = defaultdict(list)

        for img_path in distortion_images:
            parts = img_path.stem.split("_")
            if len(parts) >= 2:
                clean_id = "_".join(parts[:-1])
                distortion_map[clean_id].append(str(img_path))

        for clean_img_path in identity_folder.glob("*.jpg"):
            img_id = clean_img_path.stem
            data[identity_folder.name].append({
                "clean": str(clean_img_path),
                "img_id": img_id,
                "distortions": distortion_map.get(img_id, [])
            })

    return data

# ============== REMAINING UTILITY FUNCTIONS ==============
def generate_pairs(task_b_data):
    pos_pairs = []
    neg_pairs = []
    identities = list(task_b_data.keys())

    for identity, entries in task_b_data.items():
        for entry in entries:
            clean_img = entry["clean"]
            distortions = entry["distortions"]
            for dist_img in distortions:
                pos_pairs.append((clean_img, dist_img))
            for i in range(len(distortions)):
                for j in range(i + 1, len(distortions)):
                    pos_pairs.append((distortions[i], distortions[j]))

    while len(neg_pairs) < len(pos_pairs):
        id1, id2 = random.sample(identities, 2)
        if not task_b_data[id1] or not task_b_data[id2]:
            continue
        entry1 = random.choice(task_b_data[id1])
        entry2 = random.choice(task_b_data[id2])
        img1 = random.choice([entry1["clean"]] + entry1["distortions"])
        img2 = random.choice([entry2["clean"]] + entry2["distortions"])
        neg_pairs.append((img1, img2))

    pairs = pos_pairs + neg_pairs
    labels = [1]*len(pos_pairs) + [0]*len(neg_pairs)
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    return [p for p, _ in combined], [l for _, l in combined]

def preprocess_image(path):
    img = Image.open(path).convert('RGB').resize(IMAGE_SIZE)
    return np.asarray(img, dtype=np.float32) / 255.0


def create_dataset(pairs, labels, batch_size):
    def load_img(path):
        path = path.numpy().decode('utf-8')
        return preprocess_image(path).astype(np.float32)

    def wrapper(path):
        return tf.py_function(load_img, [path], tf.float32)

    pair1 = tf.constant([p[0] for p in pairs])
    pair2 = tf.constant([p[1] for p in pairs])
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices(((pair1, pair2), labels))
    dataset = dataset.map(
        lambda paths, label: ((wrapper(paths[0]), wrapper(paths[1])), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_real_data(train_dir, val_dir):
    print("\nLoading data using utils/siamese_data_utils.py...")
    train_data = load_task_b_structure(Path(train_dir))
    val_data = load_task_b_structure(Path(val_dir))

    train_pairs, train_labels = generate_pairs(train_data)
    val_pairs, val_labels = generate_pairs(val_data)

    train_ds = create_dataset(train_pairs, train_labels, BATCH_SIZE)
    val_ds = create_dataset(val_pairs, val_labels, BATCH_SIZE)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    return train_ds, val_ds
