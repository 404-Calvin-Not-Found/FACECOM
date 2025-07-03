import tensorflow as tf
from utils.data_utils import load_gender_classification_data
from utils.model_utils import build_gender_classifier
from config import ModelConfig, PathConfig
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU Detected: {gpus[0]}")
else:
    print("No GPU found - falling back to CPU")

class SavePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_path):
        super().__init__()
        self.output_path = output_path
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(self.epochs, self.train_loss, label='Train Loss')
        axs[0].plot(self.epochs, self.val_loss, label='Val Loss')
        axs[0].set_title('Loss Curve')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(self.epochs, self.train_acc, label='Train Accuracy')
        axs[1].plot(self.epochs, self.val_acc, label='Val Accuracy')
        axs[1].set_title('Accuracy Curve')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.tight_layout()
        fig.savefig(self.output_path)
        plt.close(fig)

# ================= TRAINING LOGIC =================
def train_gender_classifier():
    try:
        start_time = datetime.now()
        print("\nLoading gender classification data...")

        train_gen, val_gen = load_gender_classification_data()
        if not train_gen or not val_gen:
            raise ValueError("Data loading failed - check dataset paths")

        print(f"\nDataset Stats:")
        print(f"- Training samples: {train_gen.samples:,}")
        print(f"- Validation samples: {val_gen.samples:,}")
        print(f"- Class balance: {train_gen.class_indices}")

        print("\nBuilding base model...")
        model = build_gender_classifier()

        # Freeze initial layers
        base_model = model.get_layer(index=1)
        if hasattr(base_model, 'layers'):
            for layer in base_model.layers:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=ModelConfig.INITIAL_LR),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        plot_path = PathConfig.TRAINING_LOGS / 'gender_accuracy_plot.jpg'
        save_plot_cb = SavePlotCallback(str(plot_path))

        best_acc = 0.0
        best_model_weights = None

        print("\nPhase 1: Training with frozen base layers...")
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            verbose=1,
            callbacks=[save_plot_cb]
        )

        for acc in history1.history['val_accuracy']:
            if acc > best_acc and acc < 0.97:
                best_acc = acc
                best_model_weights = model.get_weights()

        print("\nUnfreezing all layers for fine-tuning...")
        if hasattr(base_model, 'layers'):
            for layer in base_model.layers:
                layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=ModelConfig.EPOCHS,
            initial_epoch=10,
            verbose=1,
            callbacks=[save_plot_cb]
        )

        for acc in history2.history['val_accuracy']:
            if acc > best_acc and acc < 0.97:
                best_acc = acc
                best_model_weights = model.get_weights()

        if best_model_weights is not None:
            model.set_weights(best_model_weights)
            model.save(str(PathConfig.GENDER_MODEL))

        duration = datetime.now() - start_time
        print(f"\nTraining completed in {duration}")
        print(f"Best val_accuracy < 97%: {best_acc:.4f}")

        return {"val_accuracy": best_acc}

    except Exception as e:
        print(f"\nTraining failed: {str(e)}", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("="*50)
    print(f"Gender Classifier Training (TF {tf.__version__})")
    print("="*50)

    history = train_gender_classifier()

    if history is None:
        print("\nTraining failed - check error messages above")
        sys.exit(1)
    else:
        print("\nTraining successful!")