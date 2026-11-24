### Face Mask Detection with Live Alert System

A real-time computer vision project that detects faces from a webcam, classifies each detected face as MASK or NO MASK, raises an audio alert for a no-mask detection, and stores detection records for dashboard-based monitoring.

# Features

- Face detection using OpenCV Haar Cascade.
- Face-mask classification using a trained CNN model.
- Real-time webcam inference.
- Visual labels and detection counts on the webcam feed.
- Audio warning for NO MASK detections.
- CSV logging of detection events.
- SQLite storage of detection category, timestamp, and confidence.
- Flask dashboard with separate MASK and NO MASK counts.
- Dashboard statistics, confidence charts, and detection history.
- Automatic dashboard refresh.

# Technology Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Flask
- SQLite
- HTML, CSS, JavaScript
- Chart.js
- Pygame (audio alert)

# Project Structure

Face-Mask-Detection-with-Live-Alert-System/
├── app.py
├── config.py
├── dataset/
│ ├── with_mask/
│ ├── without_mask/
│ └── raw/
├── models/
│ └── face_mask_detector.keras
├── results/
│ ├── evaluation_metrics.json
│ └── detection_log.csv
├── database/
│ └── face_mask_detection.db
├── static/
│ └── css/
│ └── style.css
├── templates/
│ └── index.html
└── src/
├── face_detector.py
├── mask_predictor.py
├── real_time_detection.py
├── alert_system.py
├── database.py
├── database_viewer.py
└── test_audio.py

# Setup

1. Clone the repository
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd Face-Mask-Detection-with-Live-Alert-System
2. Create and activate a virtual environment
   Windows:
   python -m venv venv
   venv\Scripts\activate
   macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate
3. Install dependencies
   If a requirements.txt file is present:
   pip install -r requirements.txt
   Otherwise, install the packages used by the project:
   pip install opencv-python tensorflow numpy flask pygame
4. Add the trained model and dataset
   Place the trained model at:
   models/face_mask_detector.keras
   Place the dataset images in the corresponding dataset/with_mask/ and dataset/without_mask/ folders if you intend to retrain the model.
   The trained model and dataset may be excluded from Git because of file size and dataset licensing. Make sure you have a permitted way to obtain them before running the application.

# Run the Application

Open two terminals from the project root.

Terminal 1 — Start the Flask dashboard:
python app.py
Open http://127.0.0.1:5000 in your browser.

Terminal 2 — Start webcam detection:
python -m src.real_time_detection
Press Q in the webcam window to quit, if supported by the running detection script.
View Database Records
python -m src.database_viewer
Results
The recorded evaluation output for the current model was:
Metric Result
Test accuracy 71.65%
Test precision 82.06%
Test recall 79.58%
Test F1-score 80.80%
Test images 1,150
Confidence threshold 0.50

These are the previously recorded evaluation results. Re-run evaluation on the final model and test split before publishing or submitting the project. Performance may vary with lighting, camera quality, face angle, occlusion, and dataset characteristics.

## Author

Name: Nishita Kanojiya

Project: Face Mask Detection with Live Alert System
