import tensorflow as tf
from utils.siamese_data_utils import load_siamese_data
from config import ModelConfig, PathConfig
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam

os.environ["TF_GPU_ALLOCATION"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU Detected: {gpus[0]}")
else:
    print("No GPU found - falling back to CPU")

@register_keras_serializable()
class L2Normalization(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(
            y_true * tf.square(y_pred) +
            (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        )
    return loss

def contrastive_accuracy(threshold=1.0):
    def acc_fn(y_true, y_pred):
        pred = tf.cast(y_pred < threshold, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(pred, y_true), tf.float32))
    acc_fn.__name__ = 'contrastive_accuracy'
    return acc_fn

class SavePlotCallback(Callback):
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
        self.train_loss.append(float(logs['loss']))
        self.val_loss.append(float(logs['val_loss']))
        self.train_acc.append(float(logs['contrastive_accuracy']))
        self.val_acc.append(float(logs['val_contrastive_accuracy']))

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(self.epochs, self.train_loss, label='Train Loss')
        axs[0].plot(self.epochs, self.val_loss, label='Val Loss')
        axs[0].legend()
        axs[0].set_title('Loss')

        axs[1].plot(self.epochs, self.train_acc, label='Train Accuracy')
        axs[1].plot(self.epochs, self.val_acc, label='Val Accuracy')
        axs[1].legend()
        axs[1].set_title('Accuracy')

        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()

def build_embedding_network():
    base_model = applications.ResNet50(
        input_shape=ModelConfig.IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    inputs = layers.Input(shape=ModelConfig.IMAGE_SIZE + (3,))
    x = base_model(inputs)
    x = layers.Dense(256, activation=None, kernel_regularizer=regularizers.l2(ModelConfig.WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation=None)(x)
    x = L2Normalization()(x)
    return models.Model(inputs, x)

def create_siamese_model(embedding):
    input_a = layers.Input(shape=ModelConfig.IMAGE_SIZE + (3,))
    input_b = layers.Input(shape=ModelConfig.IMAGE_SIZE + (3,))
    embed_a = embedding(input_a)
    embed_b = embedding(input_b)
    distance = layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1, keepdims=True))([embed_a, embed_b])
    output = layers.Activation('linear', dtype='float32')(distance)
    return models.Model(inputs=[input_a, input_b], outputs=output)

def unfreeze_layers(model, percentage=0.5):
    """Unfreeze a percentage of layers in the base model"""
    base_model = model.layers[2].layers[1]  # Get the ResNet50 base model
    total_layers = len(base_model.layers)
    num_unfrozen = int(total_layers * percentage)
    
    # Unfreeze the top layers
    for layer in base_model.layers[-num_unfrozen:]:
        layer.trainable = True
        
    print(f"\nUnfrozen {num_unfrozen}/{total_layers} layers in base model")

def train():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    train_data, val_data = load_siamese_data()
    embedding = build_embedding_network()
    model = create_siamese_model(embedding)

    plot_cb = SavePlotCallback(PathConfig.RESULTS / "training_logs" / "acc_plot.jpg")
    model_path = PathConfig.MODELS / "resnet_face_model.keras"

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        plot_cb
    ]

    # Phase 1: Train only the top layers (frozen base)
    print("\nPhase 1: Training top layers (frozen base) - 20 epochs")
    model.compile(optimizer=Adam(1e-4),
                 loss=contrastive_loss(1.0),
                 metrics=[contrastive_accuracy()])
    
    history = model.fit(train_data, 
                       epochs=20, 
                       validation_data=val_data, 
                       callbacks=callbacks, 
                       verbose=2)

    # Phase 2: Unfreeze 50% of base layers and fine-tune
    print("\nPhase 2: Fine-tuning with 50% unfrozen layers - 15 epochs")
    unfreeze_layers(model, 0.5)
    
    model.compile(optimizer=Adam(1e-5),  # Lower learning rate for fine-tuning
                 loss=contrastive_loss(1.0),
                 metrics=[contrastive_accuracy()])
    
    history = model.fit(train_data, 
                       epochs=35,  # Total epochs: 20 + 15 = 35
                       initial_epoch=20,
                       validation_data=val_data, 
                       callbacks=callbacks, 
                       verbose=2)

    # Phase 3: Unfreeze all layers for final fine-tuning
    print("\nPhase 3: Fine-tuning all layers - 15 epochs")
    unfreeze_layers(model, 1.0)  # Unfreeze all layers
    
    model.compile(optimizer=Adam(1e-6),  # Very low learning rate
                 loss=contrastive_loss(1.0),
                 metrics=[contrastive_accuracy()])
    
    history = model.fit(train_data, 
                       epochs=50,  # Total epochs: 35 + 15 = 50
                       initial_epoch=35,
                       validation_data=val_data, 
                       callbacks=callbacks, 
                       verbose=2)

    print("\nSaving best model...")
    model.save(model_path)

if __name__ == "__main__":
    print("="*50)
    print(f"Siamese Network Training with ResNet50 (TF {tf.__version__})")
    print("="*50)
    try:
        start = datetime.now()
        train()
        print(f"\nTraining completed in {datetime.now() - start}")
    except Exception as e:
        print(f"\nTraining failed: {e}", file=sys.stderr)
        raise e