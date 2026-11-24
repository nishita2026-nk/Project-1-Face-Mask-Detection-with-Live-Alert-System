
"""
Face Mask Detection
Model Evaluation Module

Evaluates the trained CNN using the independent test dataset.

Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Graphs:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Confusion Matrix
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from config import (
    RESULTS_DIR,
    MODELS_DIR,
    IMAGE_SIZE,
    CLASS_MAPPING
)


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = MODELS_DIR / "face_mask_detector.keras"

CSV_PATH = RESULTS_DIR / "dataset_splits.csv"

HISTORY_PATH = RESULTS_DIR / "training_history.json"

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"

REPORT_PATH = RESULTS_DIR / "classification_report.txt"

ACCURACY_PLOT = RESULTS_DIR / "accuracy_plot.png"

LOSS_PLOT = RESULTS_DIR / "loss_plot.png"

CONFUSION_MATRIX_PLOT = RESULTS_DIR / "confusion_matrix.png"

BATCH_SIZE = 32

CLASS_NAMES = [
    "WITHOUT MASK",
    "WITH MASK"
]

LABELS = [0, 1]


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Complete Phase 3 training first."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("\nTrained CNN model loaded successfully.")

    return model


# ==========================================
# LOAD TEST DATASET
# ==========================================

def load_test_data():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "Dataset split CSV not found. Run Phase 2 first."
        )

    dataframe = pd.read_csv(CSV_PATH)

    required_columns = {
        "image_path",
        "label",
        "split"
    }

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "Required dataset columns are missing."
        )

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    if test_df.empty:
        raise ValueError(
            "Test dataset is empty."
        )

    if set(test_df["label"].astype(int).unique()) != {0, 1}:
        raise ValueError(
            "Test dataset must contain both classes."
        )

    print("\nTest dataset loaded.")
    print("Total test images:", len(test_df))

    return test_df


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path, label):

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

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


# ==========================================
# CREATE TEST PIPELINE
# ==========================================

def create_test_dataset(test_df):

    paths = test_df["image_path"].astype(str).values

    labels = test_df["label"].astype("float32").values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate_predictions(model, test_dataset):

    print("\nGenerating predictions...")

    probabilities = model.predict(
        test_dataset,
        verbose=1
    ).ravel()

    y_true = np.concatenate([
        labels.numpy()
        for _, labels in test_dataset
    ]).astype(int)

    # Sigmoid output threshold
    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    # Evaluation metrics
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    metrics = {
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1_score": float(f1),
        "total_test_images": int(len(y_true)),
        "confidence_threshold": 0.5,
        "class_mapping": CLASS_MAPPING
    }

    print("\n========== MODEL EVALUATION ==========")

    for name, value in metrics.items():

        print(f"{name}: {value}")

    print("\nClassification Report:\n")
    print(report)

    print("\nConfusion Matrix:\n")
    print(matrix)

    return metrics, matrix, report, y_true, y_pred


# ==========================================
# SAVE METRICS
# ==========================================

def save_metrics(metrics, report):

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nEvaluation metrics saved.")
    print("Metrics:", METRICS_PATH)
    print("Report:", REPORT_PATH)


# ==========================================
# CONFUSION MATRIX GRAPH
# ==========================================

def plot_confusion_matrix(matrix):

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "Face Mask Detection - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PLOT,
        dpi=300
    )

    plt.close(fig)

    print("Confusion matrix graph saved.")


# ==========================================
# TRAINING HISTORY GRAPHS
# ==========================================

def plot_training_history():

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            "Training history not found. Run Phase 3 first."
        )

    with open(
        HISTORY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        history = json.load(file)

    # Accuracy graph
    if (
        "accuracy" not in history
        or "val_accuracy" not in history
    ):
        raise ValueError(
            "Accuracy history is incomplete."
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        history["accuracy"],
        label="Training Accuracy"
    )

    ax.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    ax.set_title(
        "Training vs Validation Accuracy"
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    plt.savefig(
        ACCURACY_PLOT,
        dpi=300
    )

    plt.close(fig)

    # Loss graph
    if (
        "loss" not in history
        or "val_loss" not in history
    ):
        raise ValueError(
            "Loss history is incomplete."
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        history["loss"],
        label="Training Loss"
    )

    ax.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    ax.set_title(
        "Training vs Validation Loss"
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT,
        dpi=300
    )

    plt.close(fig)

    print("Accuracy graph saved.")
    print("Loss graph saved.")


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n================================")
    print("FACE MASK MODEL EVALUATION")
    print("================================")

    model = load_trained_model()

    test_df = load_test_data()

    test_dataset = create_test_dataset(
        test_df
    )

    metrics, matrix, report, _, _ = evaluate_predictions(
        model,
        test_dataset
    )

    save_metrics(
        metrics,
        report
    )

    plot_confusion_matrix(
        matrix
    )

    plot_training_history()

    print("\n================================")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("================================")

    print("\nGenerated files:")

    print(METRICS_PATH)
    print(REPORT_PATH)
    print(CONFUSION_MATRIX_PLOT)
    print(ACCURACY_PLOT)
    print(LOSS_PLOT)


if __name__ == "__main__":

    main()
