
import sqlite3
from datetime import datetime
from config import DATABASE_DIR


class DatabaseManager:

    def __init__(self):

        DATABASE_DIR.mkdir(parents=True, exist_ok=True)

        self.db_path = DATABASE_DIR / "face_mask_detection.db"

        self.initialize_database()


    def initialize_database(self):

        connection = sqlite3.connect(self.db_path)

        try:

            cursor = connection.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    timestamp TEXT NOT NULL,

                    detection TEXT NOT NULL,

                    confidence REAL NOT NULL

                )
            """)

            connection.commit()

            print("SQLite database initialized successfully!")

        finally:

            connection.close()


    def insert_detection(self, detection, confidence):

        detection = detection.upper()

        if detection not in ("MASK", "NO MASK"):
            raise ValueError("Invalid detection category")

        connection = sqlite3.connect(self.db_path)

        try:

            cursor = connection.cursor()

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            cursor.execute("""
                INSERT INTO detections
                (timestamp, detection, confidence)

                VALUES (?, ?, ?)
            """, (
                timestamp,
                detection,
                float(confidence)
            ))

            connection.commit()

            print(f"{detection} detection saved to SQLite!")

        finally:

            connection.close()


    def get_all_detections(self):

        connection = sqlite3.connect(self.db_path)

        try:

            cursor = connection.cursor()

            cursor.execute("""
                SELECT id, timestamp, detection, confidence

                FROM detections

                ORDER BY id DESC
            """)

            return cursor.fetchall()

        finally:

            connection.close()


    def get_statistics(self):

        connection = sqlite3.connect(self.db_path)

        try:

            cursor = connection.cursor()

            cursor.execute("""
                SELECT

                    COUNT(*),

                    SUM(CASE
                        WHEN detection = 'MASK' THEN 1
                        ELSE 0
                    END),

                    SUM(CASE
                        WHEN detection = 'NO MASK' THEN 1
                        ELSE 0
                    END),

                    AVG(confidence),

                    AVG(CASE
                        WHEN detection = 'MASK' THEN confidence
                    END),

                    AVG(CASE
                        WHEN detection = 'NO MASK' THEN confidence
                    END)

                FROM detections
            """)

            result = cursor.fetchone()

            return {

                "total_face_detections": result[0] or 0,

                "mask_detections": result[1] or 0,

                "no_mask_detections": result[2] or 0,

                "average_confidence": round(result[3], 2)
                if result[3] is not None else 0,

                "average_mask_confidence": round(result[4], 2)
                if result[4] is not None else 0,

                "average_no_mask_confidence": round(result[5], 2)
                if result[5] is not None else 0

            }

        finally:

            connection.close()


if __name__ == "__main__":

    db = DatabaseManager()

    print("\nDATABASE STATISTICS")

    print(db.get_statistics())

    print("\nDatabase upgrade completed!")
