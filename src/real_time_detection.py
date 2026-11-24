
import cv2
import time

from config import RESULTS_DIR
from src.face_detector import FaceDetector
from src.mask_predictor import MaskPredictor
from src.alert_system import AlertSystem


def draw_label(frame, text, x, y, color):

    cv2.rectangle(
        frame,
        (x, y - 30),
        (x + 220, y),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def draw_dashboard(frame, total, masked, unmasked, fps):

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 85),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "FACE MASK DETECTION SYSTEM",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    stats = (
        f"Faces: {total} | "
        f"Mask: {masked} | "
        f"No Mask: {unmasked} | "
        f"FPS: {fps:.1f}"
    )

    cv2.putText(
        frame,
        stats,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


def start_detection():

    detector = FaceDetector()
    predictor = MaskPredictor()
    alert_system = AlertSystem()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        alert_system.close()

        raise RuntimeError("Unable to access webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous_time = time.time()

    print("Face Mask Detection Started!")
    print("Press Q to quit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print("Unable to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            faces = detector.detect_faces(frame)

            # Detection counters
            mask_count = 0
            no_mask_count = 0

            # Confidence lists
            mask_confidences = []
            no_mask_confidences = []

            # Process every detected face
            for (x, y, w, h) in faces:

                face_roi = frame[y:y+h, x:x+w]

                try:

                    label, confidence = predictor.predict(face_roi)

                except (ValueError, cv2.error) as error:

                    print("Prediction error:", error)
                    continue

                if label == 1:

                    mask_count += 1

                    mask_confidences.append(confidence)

                    color = (0, 200, 0)

                    status = "MASK"

                else:

                    no_mask_count += 1

                    no_mask_confidences.append(confidence)

                    color = (0, 0, 255)

                    status = "NO MASK"

                # Draw face rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                # Draw prediction label
                draw_label(
                    frame,
                    f"{status}: {confidence:.1f}%",
                    x,
                    max(y, 35),
                    color
                )

            # =================================
            # SAVE MASK DETECTIONS
            # =================================

            if mask_confidences:

                alert_system.trigger_mask(
                    max(mask_confidences)
                )

            # =================================
            # SAVE NO MASK DETECTIONS
            # =================================

            if no_mask_confidences:

                # Warning banner
                cv2.rectangle(
                    frame,
                    (10, 95),
                    (450, 145),
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    "WARNING: NO MASK DETECTED!",
                    (20, 128),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                alert_system.trigger_no_mask(
                    max(no_mask_confidences)
                )

            # Calculate FPS
            current_time = time.time()

            fps = 1 / max(
                current_time - previous_time,
                0.001
            )

            previous_time = current_time

            # Display webcam dashboard
            draw_dashboard(
                frame,
                len(faces),
                mask_count,
                no_mask_count,
                fps
            )

            cv2.imshow(
                "Face Mask Detection System",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        alert_system.close()

        print("Detection stopped.")


if __name__ == "__main__":

    try:

        start_detection()

    except (RuntimeError, FileNotFoundError) as error:

        print("Error:", error)
