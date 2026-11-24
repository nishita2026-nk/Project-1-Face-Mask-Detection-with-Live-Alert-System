
"""
Face Mask Detection
Haar Cascade Face Detection Module
"""

import cv2


class FaceDetector:
    """Detect human faces using Haar Cascade."""

    def __init__(self, scale_factor=1.1, min_neighbors=5):

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                "Unable to load Haar Cascade classifier."
            )

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect_faces(self, frame):
        """
        Detect faces in a BGR webcam frame.

        Returns:
            List of (x, y, width, height) bounding boxes.
        """

        if frame is None or frame.size == 0:
            return []

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(40, 40)
        )

        return [tuple(map(int, face)) for face in faces]
