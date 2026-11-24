
"""
Face Mask Detection
CNN Model Architecture

Input: 128 x 128 x 3
Output:
    0 = WITHOUT MASK
    1 = WITH MASK
"""

import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

from config import IMAGE_SIZE, IMAGE_CHANNELS


def build_model(learning_rate=0.001):
    """
    Build and compile a CNN binary classification model.

    Returns:
        Compiled TensorFlow Keras model.
    """

    # Input layer
    inputs = layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            IMAGE_CHANNELS
        ),
        name="input_image"
    )

    # Data augmentation
    # Active during training, inactive during inference.
    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(
                height_factor=0.10,
                width_factor=0.10
            )
        ],
        name="data_augmentation"
    )

    x = augmentation(inputs)

    # First convolution block
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(0.0001)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolution block
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(0.0001)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Third convolution block
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(0.0001)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Fourth convolution block
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Feature extraction
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.4)(x)

    # Binary classification output
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="mask_prediction"
    )(x)

    # Create model
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Face_Mask_Detection_CNN"
    )

    # Compile model
    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


if __name__ == "__main__":

    model = build_model()

    print("\nCNN MODEL ARCHITECTURE")
    model.summary()

    print("\nModel successfully created and compiled.")
