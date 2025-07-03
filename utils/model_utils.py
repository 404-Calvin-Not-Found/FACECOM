from tensorflow.keras import layers, models, applications
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from config import ModelConfig


@register_keras_serializable()
class L2Normalization(layers.Layer):
    """Custom L2 normalization layer with proper serialization."""
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()


def build_gender_classifier():
    """Build model for Task A (Gender Classification)"""
    print("\nBuilding gender classifier...")

    base_model = applications.MobileNetV2(
        input_shape=ModelConfig.IMAGE_SIZE + (ModelConfig.CHANNELS,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=ModelConfig.IMAGE_SIZE + (ModelConfig.CHANNELS,)),
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ModelConfig.INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    return model


def build_embedding_network():
    """Build the shared embedding network for Task B"""
    print("\nBuilding embedding network...")

    base_model = applications.MobileNetV2(
        input_shape=ModelConfig.IMAGE_SIZE + (ModelConfig.CHANNELS,),
        include_top=False,
        weights='imagenet'
    )

    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = models.Sequential([
        layers.Input(shape=ModelConfig.IMAGE_SIZE + (ModelConfig.CHANNELS,)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(ModelConfig.EMBEDDING_SIZE, activation=None),
        L2Normalization()
    ])

    return model


def build_siamese_network(embedding_network):
    """Build Siamese network for Task B"""
    print("\nBuilding Siamese network...")

    input1 = layers.Input(shape=ModelConfig.IMAGE_SIZE + (ModelConfig.CHANNELS,))
    input2 = layers.Input(shape=ModelConfig.IMAGE_SIZE + (ModelConfig.CHANNELS,))

    embedding1 = embedding_network(input1)
    embedding2 = embedding_network(input2)

    distance = layers.Lambda(
        lambda embeddings: tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)
    )([embedding1, embedding2])

    output = layers.Dense(1, activation='sigmoid')(distance)

    siamese_model = models.Model(inputs=[input1, input2], outputs=output)

    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(ModelConfig.INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return siamese_model
