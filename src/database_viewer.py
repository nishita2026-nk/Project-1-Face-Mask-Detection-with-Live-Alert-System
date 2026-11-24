
from src.database import DatabaseManager


def display_detection_history():

    db = DatabaseManager()
    records = db.get_all_detections()

    print("\n========== DETECTION HISTORY ==========\n")

    if not records:
        print("No detection records found.")
        return

    print(
        f"{'ID':<6}"
        f"{'Timestamp':<23}"
        f"{'Detection':<15}"
        f"{'Confidence':<12}"
    )

    print("-" * 60)

    for record in records:
        record_id, timestamp, detection, confidence = record

        print(
            f"{record_id:<6}"
            f"{timestamp:<23}"
            f"{detection:<15}"
            f"{confidence:<12.2f}%"
        )


def display_statistics():

    db = DatabaseManager()
    stats = db.get_statistics()

    print("\n========== DETECTION STATISTICS ==========\n")

    print("Total Face Detections:",
          stats["total_face_detections"])

    print("MASK Detections:",
          stats["mask_detections"])

    print("NO MASK Detections:",
          stats["no_mask_detections"])

    print("Average Confidence:",
          f'{stats["average_confidence"]:.2f}%')

    print("Average MASK Confidence:",
          f'{stats["average_mask_confidence"]:.2f}%')

    print("Average NO MASK Confidence:",
          f'{stats["average_no_mask_confidence"]:.2f}%')


if __name__ == "__main__":

    display_detection_history()
    display_statistics()
