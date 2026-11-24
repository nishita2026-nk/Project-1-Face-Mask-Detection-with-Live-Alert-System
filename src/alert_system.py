
import csv
import time
import numpy as np
import pygame

from datetime import datetime

from config import RESULTS_DIR, ALERT_COOLDOWN
from src.database import DatabaseManager


class AlertSystem:

    def __init__(self):

        self.last_detection_time = {
            "MASK": 0,
            "NO MASK": 0
        }

        self.log_file = RESULTS_DIR / "detection_log.csv"

        self.audio_available = False
        self.alert_sound = None

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.database = DatabaseManager()

        # Initialize audio system
        try:

            pygame.mixer.init(
                frequency=44100,
                size=-16,
                channels=1,
                buffer=512
            )

            sample_rate = 44100
            duration = 500
            frequency = 1000

            samples = np.arange(
                int(sample_rate * duration / 1000)
            )

            wave = 0.5 * np.sin(
                2 * np.pi * frequency * samples / sample_rate
            )

            audio = (wave * 32767).astype(np.int16)

            self.alert_sound = pygame.mixer.Sound(
                buffer=audio.tobytes()
            )

            self.alert_sound.set_volume(1.0)

            self.audio_available = True

            print("Audio system initialized!")

        except Exception as error:

            print("Audio initialization failed:", error)


        # Create CSV file if it does not exist
        if not self.log_file.exists():

            with open(
                self.log_file,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Date",
                    "Time",
                    "Detection",
                    "Confidence"
                ])


    def process_detection(self, detection, confidence):

        detection = detection.upper()

        if detection not in ("MASK", "NO MASK"):
            return False

        current_time = time.time()

        # Separate cooldown for each detection category
        if (
            current_time - self.last_detection_time[detection]
            < ALERT_COOLDOWN
        ):
            return False

        print(f"\n{detection} DETECTED: {confidence:.2f}%")

        # Audio alert only for NO MASK
        if detection == "NO MASK":

            print("!!! NO MASK ALERT !!!")

            if self.audio_available:

                try:

                    self.alert_sound.play()

                    print("AUDIO ALERT PLAYED!")

                except pygame.error as error:

                    print("Audio error:", error)


        # Save to CSV
        self.log_detection(detection, confidence)

        # Save to SQLite
        try:

            self.database.insert_detection(
                detection,
                confidence
            )

        except Exception as error:

            print("Database logging error:", error)


        self.last_detection_time[detection] = current_time

        return True


    def trigger_no_mask(self, confidence):

        return self.process_detection("NO MASK", confidence)


    def trigger_mask(self, confidence):

        return self.process_detection("MASK", confidence)


    def log_detection(self, detection, confidence):

        now = datetime.now()

        with open(
            self.log_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                detection,
                round(float(confidence), 2)
            ])

        print("CSV log updated!")


    def close(self):

        if self.audio_available:

            pygame.mixer.quit()

        print("Alert system closed!")


if __name__ == "__main__":

    alert = AlertSystem()

    print("Alert system initialized successfully!")

    alert.close()
