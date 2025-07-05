import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from pathlib import Path
from utils.siamese_data_utils import (
    build_embedding_network,
    create_siamese_model,
    contrastive_loss,
    contrastive_accuracy,
    unfreeze_layers,
    load_real_data
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import time

# ============== CONFIGURATION ==============
MARGIN = 0.8  # Slightly lowered for better robustness
MODEL_SAVE_PATH = "models/face_recognizer.keras"
TRAIN_DIR = "D:/FACECOM/datasets/Task_B/train"
VAL_DIR = "D:/FACECOM/datasets/Task_B/val"

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

# ============== CALLBACK FOR PROGRESS BAR + ETA ==============
class EpochProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n\033[92mEpoch {epoch + 1} started\033[0m")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        mins, secs = divmod(elapsed, 60)
        print(f"Epoch complete in {int(mins)}m {int(secs)}s\n")

# ============== TRAINING SCRIPT ==============
def train():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    train_ds, val_ds = load_real_data(TRAIN_DIR, VAL_DIR)

    embedding = build_embedding_network()
    model = create_siamese_model(embedding)

    callbacks = [
        SaveBestModelCallback(model, MODEL_SAVE_PATH, target_accuracy=0.97),
        ReduceLROnPlateau(monitor='val_contrastive_accuracy', mode='max', patience=2, factor=0.5, verbose=1),
        EarlyStopping(monitor='val_contrastive_accuracy', patience=5, restore_best_weights=True, mode='max'),
        LivePlotCallback(),
        EpochProgressBar()
    ]

    print("\nPhase 1: Train frozen base layers")
    model.compile(optimizer=Adam(1e-4), loss=contrastive_loss(MARGIN), metrics=[contrastive_accuracy(MARGIN)])
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks, verbose=2)

    print("\nPhase 2: Fine-tune last 50% of base model")
    unfreeze_layers(model, 0.5)
    model.compile(optimizer=Adam(5e-5), loss=contrastive_loss(MARGIN), metrics=[contrastive_accuracy(MARGIN)])
    model.fit(train_ds, validation_data=val_ds, epochs=30, initial_epoch=15, callbacks=callbacks, verbose=2)

    print("\nPhase 3: Fine-tune entire base model")
    unfreeze_layers(model, 1.0)
    model.compile(optimizer=Adam(1e-5), loss=contrastive_loss(MARGIN), metrics=[contrastive_accuracy(MARGIN)])
    model.fit(train_ds, validation_data=val_ds, epochs=45, initial_epoch=30, callbacks=callbacks, verbose=2)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("==============================")
    print(" Training Siamese Model ")
    print("==============================")
    train()
