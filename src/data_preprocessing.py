
"""
Face Mask Detection
Dataset Loading and Preprocessing Module

Responsibilities:
1. Validate dataset directories.
2. Load image paths.
3. Detect corrupted images.
4. Encode class labels.
5. Create stratified train/validation/test splits.
6. Resize images.
7. Convert BGR to RGB.
8. Normalize pixel values.
9. Save dataset split information.
"""

import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from config import (
    WITH_MASK_DIR,
    WITHOUT_MASK_DIR,
    RESULTS_DIR,
    IMAGE_SIZE,
    IMAGE_EXTENSIONS,
    CLASS_MAPPING,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED
)


def validate_dataset():
    """Check whether both dataset class folders exist."""

    required_folders = [
        WITH_MASK_DIR,
        WITHOUT_MASK_DIR
    ]

    for folder in required_folders:
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        if not folder.is_dir():
            raise NotADirectoryError(
                f"Expected a directory: {folder}"
            )

    print("Dataset directories validated successfully.")


def collect_image_paths():
    """
    Collect valid image paths and corresponding labels.

    Label 0 = Without Mask
    Label 1 = With Mask
    """

    validate_dataset()

    dataset_records = []
    corrupted_count = 0

    class_directories = [
        (WITHOUT_MASK_DIR, "without_mask"),
        (WITH_MASK_DIR, "with_mask")
    ]

    for folder, class_name in class_directories:

        label = CLASS_MAPPING[class_name]

        image_paths = [
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(
            f"\nChecking {class_name}: "
            f"{len(image_paths)} images found"
        )

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            if image is None:
                corrupted_count += 1
                print(f"Skipping corrupted image: {image_path}")
                continue

            if image.size == 0:
                corrupted_count += 1
                continue

            dataset_records.append({
                "image_path": str(image_path.resolve()),
                "class_name": class_name,
                "label": label
            })

    if not dataset_records:
        raise ValueError(
            "No valid images found in the dataset."
        )

    dataframe = pd.DataFrame(dataset_records)

    print("\nDataset collection completed.")
    print("Valid images:", len(dataframe))
    print("Corrupted images:", corrupted_count)

    return dataframe


def split_dataset(dataframe):
    """
    Create stratified training, validation and testing splits.

    The same image path cannot appear in multiple splits.
    """

    if len(dataframe) == 0:
        raise ValueError("Dataset is empty.")

    if not np.isclose(
        TRAIN_SPLIT + VALIDATION_SPLIT + TEST_SPLIT,
        1.0
    ):
        raise ValueError("Dataset split percentages must sum to 1.")

    if dataframe["label"].nunique() != 2:
        raise ValueError(
            "Both mask and no-mask classes must contain valid images."
        )

    # First separate the test dataset.
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=dataframe["label"]
    )

    # Calculate validation fraction of remaining data.
    validation_fraction = (
        VALIDATION_SPLIT /
        (TRAIN_SPLIT + VALIDATION_SPLIT)
    )

    # Separate training and validation datasets.
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=validation_fraction,
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"]
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    validation_df["split"] = "validation"
    test_df["split"] = "test"

    final_df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True
    )

    # Check that no image occurs in multiple splits.
    if final_df["image_path"].duplicated().any():
        raise ValueError(
            "Duplicate image paths detected across dataset records."
        )

    print("\nDataset splitting completed.")

    for split_name in ["train", "validation", "test"]:

        split_data = final_df[
            final_df["split"] == split_name
        ]

        print(
            f"{split_name.capitalize()} images: "
            f"{len(split_data)}"
        )

    return final_df


def preprocess_image(image_path):
    """
    Read and preprocess one image.

    Steps:
    1. Read image.
    2. Convert BGR to RGB.
    3. Resize to 128x128.
    4. Normalize pixels to [0, 1].
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32) / 255.0

    return image


def save_dataset_splits(dataframe):
    """Save the dataset split information as a CSV file."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RESULTS_DIR / "dataset_splits.csv"

    dataframe.to_csv(
        output_path,
        index=False
    )

    print("\nDataset split file saved at:")
    print(output_path)


def display_dataset_summary(dataframe):
    """Display class distribution for every dataset split."""

    print("\n========== DATASET SUMMARY ==========")

    summary = pd.crosstab(
        dataframe["split"],
        dataframe["class_name"]
    )

    print(summary)

    print("\nTotal images:", len(dataframe))


def main():

    print("\nFACE MASK DETECTION")
    print("DATASET PREPROCESSING STARTED")

    # Collect images
    dataframe = collect_image_paths()

    # Split dataset
    dataframe = split_dataset(dataframe)

    # Save split information
    save_dataset_splits(dataframe)

    # Display statistics
    display_dataset_summary(dataframe)

    # Verify image preprocessing on one sample.
    sample_path = dataframe.iloc[0]["image_path"]

    sample_image = preprocess_image(sample_path)

    print("=============================================================================")
    print("\nSample image preprocessing successful.")
    print("Image shape:", sample_image.shape)
    print("Image data type:", sample_image.dtype)
    print("Minimum pixel value:", sample_image.min())
    print("Maximum pixel value:", sample_image.max())

    print("\nDATASET PREPROCESSING COMPLETED SUCCESSFULLY.")
    print("=============================================================================")
if __name__ == "__main__":
    main()
