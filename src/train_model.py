
"""
Face Mask Detection
CNN Training Module

Loads the dataset split created in Phase 2,
trains the CNN and saves the best model.
"""

import json
import pandas as pd
import tensorflow as tf

from pathlib import Path

from config import (
    RESULTS_DIR,
    MODELS_DIR,
    IMAGE_SIZE,
    RANDOM_SEED,
    CLASS_MAPPING
)

from src.model_builder import build_model


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001

CSV_PATH = RESULTS_DIR / "dataset_splits.csv"

MODEL_PATH = MODELS_DIR / "face_mask_detector.keras"

HISTORY_PATH = RESULTS_DIR / "training_history.json"

CONFIG_PATH = RESULTS_DIR / "training_config.json"

CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================
# REPRODUCIBILITY
# ==========================================

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ==========================================
# LOAD DATASET SPLITS
# ==========================================

def load_dataset_splits():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset split CSV not found: {CSV_PATH}\n"
            "Run Phase 2 preprocessing first."
        )

    dataframe = pd.read_csv(CSV_PATH)

    required_columns = {
        "image_path",
        "class_name",
        "label",
        "split"
    }

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "Dataset CSV is missing required columns."
        )

    if dataframe.empty:
        raise ValueError("Dataset CSV is empty.")

    if dataframe["image_path"].duplicated().any():
        raise ValueError(
            "Duplicate image paths found in dataset CSV."
        )

    if not set(dataframe["split"].unique()).issubset(
        {"train", "validation", "test"}
    ):
        raise ValueError("Invalid dataset split name found.")

    if set(dataframe["split"].unique()) != {
        "train", "validation", "test"
    }:
        raise ValueError(
            "Training, validation and test splits are required."
        )

    expected_mapping = CLASS_MAPPING

    for _, row in dataframe.iterrows():

        expected_label = expected_mapping.get(
            row["class_name"]
        )

        if expected_label is None or int(row["label"]) != expected_label:
            raise ValueError(
                f"Invalid label mapping for {row['image_path']}"
            )

    print("\nDataset split CSV loaded successfully.")

    for split in ["train", "validation", "test"]:

        split_df = dataframe[
            dataframe["split"] == split
        ]

        print(f"{split.capitalize()} images: {len(split_df)}")

        if split_df.empty:
            raise ValueError(
                f"{split} dataset is empty."
            )

        if split_df["label"].nunique() != 2:
            raise ValueError(
                f"Both classes must be present in {split} data."
            )

    return dataframe


# ==========================================
# IMAGE LOADING AND PREPROCESSING
# ==========================================

def load_and_preprocess_image(image_path, label):

    image_content = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_content,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.float32)

    return image, label


# ==========================================
# CREATE TF.DATA PIPELINE
# ==========================================

def create_dataset(dataframe, training=False):

    paths = dataframe["image_path"].astype(str).values

    labels = dataframe["label"].astype("float32").values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ==========================================
# SAVE TRAINING CONFIGURATION
# ==========================================

def save_training_config():

    configuration = {
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "class_mapping": CLASS_MAPPING,
        "optimizer": "Adam",
        "loss": "binary_crossentropy",
        "architecture": "CNN",
        "augmentation": [
            "Horizontal Flip",
            "Random Rotation",
            "Random Zoom",
            "Random Translation"
        ]
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:

        json.dump(
            configuration,
            file,
            indent=4
        )


# ==========================================
# SAVE CLASS MAPPING
# ==========================================

def save_class_mapping():

    with open(
        CLASS_MAPPING_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            CLASS_MAPPING,
            file,
            indent=4
        )


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def train_model():

    print("\n================================")
    print("FACE MASK DETECTION CNN TRAINING")
    print("================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataframe = load_dataset_splits()

    # Separate datasets
    train_df = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation_df = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    # Create TensorFlow datasets
    train_dataset = create_dataset(
        train_df,
        training=True
    )

    validation_dataset = create_dataset(
        validation_df,
        training=False
    )

    # Test dataset is intentionally not used during fitting.
    print("\nTraining pipeline created.")
    print("Validation pipeline created.")
    print("Test dataset reserved for Phase 4 evaluation.")

    # Build CNN
    model = build_model(
        learning_rate=LEARNING_RATE
    )

    print("\nCNN architecture:")
    model.summary()

    # Training callbacks
    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.000001,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            str(RESULTS_DIR / "training_log.csv")
        )
    ]

    # Start training
    print("\nStarting CNN training...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save best model with restored best weights.
    model.save(MODEL_PATH)

    # Save training history
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    # Save configuration and class mapping
    save_training_config()
    save_class_mapping()

    # Training summary
    print("\n================================")
    print("CNN TRAINING COMPLETED")
    print("================================")

    print("\nBest validation accuracy:")

    best_accuracy = max(
        history.history["val_accuracy"]
    )

    print(f"{best_accuracy * 100:.2f}%")

    print("\nSaved files:")

    print("Model:", MODEL_PATH)
    print("History:", HISTORY_PATH)
    print("Configuration:", CONFIG_PATH)
    print("Class mapping:", CLASS_MAPPING_PATH)

    print(
        "\nTest images reserved for independent evaluation:",
        len(test_df)
    )

    print("\nProceed to Phase 4 for model evaluation.")


if __name__ == "__main__":

    train_model()
