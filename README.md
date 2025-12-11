# SmartAttendanceSystem

This project implements a smart attendance system using face recognition. It leverages computer vision libraries like MediaPipe and OpenCV, and uses ONNX models for efficient face detection and embedding generation.

## Features

- **Face Registration:** Register students by capturing their face via webcam and saving their face embeddings.
- **Face Recognition:** Recognize registered students from webcam feed and mark their attendance.
- **Attendance Logging:** Records attendance in a CSV file with student name and timestamp.

## Technologies Used

- Python 3.11+
- `mediapipe`: For face detection and landmark extraction.
- `numpy`: For numerical operations, especially with embeddings.
- `onnxruntime`: For running optimized ONNX (Open Neural Network Exchange) models for face detection (SCRFD) and embedding generation (Mobile-ArcFace/MobileFaceNet).
- `opencv-python`: For webcam access, image processing, and drawing annotations.
- `uv`: A fast Python package installer and resolver.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/SmartAttendanceSystem.git
    cd SmartAttendanceSystem
    ```

2.  **Install `uv` (if not already installed):**

    ```bash
    pip install uv
    ```

    (Or follow instructions on the `uv` website for your OS)

3.  **Install dependencies using `uv`:**

    ```bash
    uv sync
    ```

4.  **Download ONNX Models:**
    Place your `scrfd.onnx` (a small face detection model) and `arcface_mobile.onnx` (a face embedding model) into the `models/` directory. These models are not included in the repository due to their size, and can be obtained from various open-source computer vision model hubs.

    Example:

    ```
    SmartAttendanceSystem/
    ├── models/
    │   ├── scrfd.onnx
    │   └── arcface_mobile.onnx
    └── ...
    ```

## Usage

### 1. Register a Student

To register a new student, run the `register.py` script:

```bash
uv run register.py
```

Follow the prompts to enter the student's name and capture their face using your webcam. Press `SPACE` to capture the face when it's clearly visible.

### 2. Start Attendance Recognition

To start the attendance system, run the `recognize.py` script:

```bash
uv run recognize.py
```

The system will use your webcam to detect and recognize faces. If a registered face is recognized, their attendance will be marked in `attendance.csv`.

**Note:**

- Ensure proper lighting and a clear view of the face for optimal performance.
- Adjust the `THRESHOLD` in `recognize.py` if needed (a lower value means stricter recognition).

## Project Structure

- `register.py`: Script for registering new student faces.
- `recognize.py`: Main script for real-time face recognition and attendance marking.
- `utils.py`: Contains core computer vision functionalities, including face detection (SCRFD/MediaPipe), face alignment, and embedding generation using ONNX models.
- `models/`: Directory to store the ONNX face detection and embedding models.
- `encodings/`: Directory where registered face embeddings (`.npy` files) are stored.
- `attendance.csv`: CSV file to log attendance records.
- `pyproject.toml`: Project metadata and dependency definitions (PEP 621).
- `uv.lock`: Lock file for `uv` package manager, ensuring reproducible environments.
- `Frontend/index.html`: (Unused in current Python scripts, but present in the directory structure).

## Development Conventions

- Python 3.11+ is required.
- Dependencies are managed using `uv`.
- Face detection uses SCRFD (if available) or MediaPipe.
- Face embeddings are generated using a Mobile-ArcFace/MobileFaceNet ONNX model.
- Face encodings are stored as NumPy array (`.npy`) files.
