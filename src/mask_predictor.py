
"""
Face Mask Detection
CNN Mask Prediction Module
"""

import cv2
import numpy as np
import tensorflow as tf

from config import (
    MODELS_DIR,
    IMAGE_SIZE,
    CLASS_MAPPING
)


class MaskPredictor:
    """Predict whether a detected face has a mask."""

    def __init__(self, threshold=0.5):

        self.model_path = (
            MODELS_DIR / "face_mask_detector.keras"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}\n"
                "Complete CNN training first."
            )

        self.model = tf.keras.models.load_model(
            self.model_path
        )

        self.threshold = threshold

        self.without_mask_label = CLASS_MAPPING["without_mask"]
        self.with_mask_label = CLASS_MAPPING["with_mask"]

        print("CNN mask prediction model loaded.")

    def preprocess_face(self, face):

        if face is None or face.size == 0:
            raise ValueError("Invalid face image.")

        # OpenCV BGR to RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        face = cv2.resize(
            face,
            IMAGE_SIZE
        )

        # Normalize
        face = face.astype(
            np.float32
        ) / 255.0

        # Add batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        return face

    def predict(self, face):
        """
        Returns:
            predicted label, confidence percentage
        """

        processed_face = self.preprocess_face(face)

        prediction = self.model(
            processed_face,
            training=False
        ).numpy().reshape(-1)[0]

        prediction = float(prediction)

        if not np.isfinite(prediction):
            raise ValueError(
                "Model returned an invalid prediction."
            )

        if prediction >= self.threshold:

            label = self.with_mask_label
            confidence = prediction

        else:

            label = self.without_mask_label
            confidence = 1.0 - prediction

        return label, confidence * 100
