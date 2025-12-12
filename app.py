"""
app.py
- Main Flask application for the Smart Attendance System.
- Handles serving HTML pages and API endpoints for face registration and recognition.
"""

import base64
import csv
import datetime
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from utils import FaceRecognitionUtils

# --- Configuration & Global State ---
app = Flask(__name__)

# Initialize utilities (loads ONNX models)
# Ensure models folder exists or handle error gracefully
try:
    face_utils = FaceRecognitionUtils()
except Exception as e:
    print(f"CRITICAL: Failed to initialize FaceRecognitionUtils: {e}")
    face_utils = None

# Recognition State
known_face_encodings = {}
last_seen_timestamps = {}  # {name: timestamp}
ATTENDANCE_CSV_FILE = "attendance.csv"
RECOGNITION_THRESHOLD = 0.5
ATTENDANCE_COOLDOWN_SECONDS = 60

def load_face_encodings():
    """
    Loads all face encodings from the encodings directory into memory.
    """
    global known_face_encodings
    if face_utils:
        try:
            known_face_encodings = face_utils.load_all_encodings()
            print(f"[System] Loaded {len(known_face_encodings)} encodings.")
        except Exception as e:
            print(f"[Error] Loading encodings: {e}")
            known_face_encodings = {}

# Initial load
load_face_encodings()

# --- Helper Functions ---

def mark_attendance(name: str) -> tuple[bool, str]:
    """
    Marks attendance for a user in the CSV file with the current UTC timestamp.
    
    Args:
        name (str): The name of the person.

    Returns:
        tuple[bool, str]: (Success status, Message).
    """
    try:
        # Check if we can write to file
        with open(ATTENDANCE_CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, datetime.datetime.utcnow().isoformat()])
        print(f"[Attendance] Marked for {name}")
        return True, "Success"
    except PermissionError:
        print(f"[Error] Permission denied for {ATTENDANCE_CSV_FILE}. Is it open?")
        return False, "File access error. Close CSV if open."
    except Exception as e:
        print(f"[Error] Marking attendance: {e}")
        # Return a user-friendly error if possible, or just the string
        return False, "Internal Error"

def decode_base64_to_image(data_url: str) -> np.ndarray:
    """
    Decodes a base64 data URL into an OpenCV image (BGR).
    
    Args:
        data_url (str): Base64 string (e.g. "data:image/jpeg;base64,...").

    Returns:
        np.ndarray: BGR Image or None if decoding fails.
    """
    try:
        if "," in data_url:
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
            
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[Error] Decoding image: {e}")
        return None

# --- Routes ---

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/register')
def register_page():
    """Renders the face registration page."""
    return render_template('register.html')

@app.route('/recognize')
def recognize_page():
    """
    Renders the face recognition/attendance page.
    Refreshes encodings on load to ensure new registrations are active.
    """
    load_face_encodings()
    return render_template('recognize.html')

@app.route('/docs')
def docs_page():
    """Renders the documentation page."""
    return render_template('docs/index.html')

@app.route('/api/register_capture', methods=['POST'])
def api_register_capture():
    """
    API call to register a new face from a captured image.
    Expects JSON: { "name": "...", "image": "base64..." }
    """
    if not face_utils:
        return jsonify({"success": False, "message": "Server not initialized."})

    data = request.json
    name = data.get('name')
    image_data = data.get('image')
    
    if not name or not image_data:
        return jsonify({"success": False, "message": "Name and Image are required."})

    frame = decode_base64_to_image(image_data)
    if frame is None:
        return jsonify({"success": False, "message": "Invalid image data."})

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_bboxes = face_utils.detect_faces_rgb(rgb_frame)
    if not face_bboxes:
        return jsonify({"success": False, "message": "No face detected. Please try again."})

    # Pick the largest face (Assumption: User is close to camera)
    # bbox: [x1, y1, x2, y2] -> Area = (x2-x1)*(y2-y1)
    sorted_bboxes = sorted(face_bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    target_box = sorted_bboxes[0]
    
    # Preprocess and Align
    face_tensor = face_utils.preprocess_face(rgb_frame, target_box)
    if face_tensor is None:
        return jsonify({"success": False, "message": "Face alignment failed."})

    # Generate Embedding
    face_embedding = face_utils.get_embedding_from_face_tensor(face_tensor)
    
    # Save to disk
    face_utils.save_encoding(name, face_embedding)
    
    # Reload to update memory
    load_face_encodings()
    
    return jsonify({"success": True, "message": f"Successfully registered {name}!"})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """
    API call to recognize faces in a frame and mark attendance.
    Expects JSON: { "image": "base64..." }
    """
    if not face_utils:
        return jsonify({"success": False, "matches": []})

    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"success": False, "matches": []})

    frame = decode_base64_to_image(image_data)
    if frame is None:
        return jsonify({"success": False, "matches": []})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_bboxes = face_utils.detect_faces_rgb(rgb_frame)
    
    matches_payload = []
    attendance_error = None
    
    known_names = list(known_face_encodings.keys())
    
    if known_names:
        # Stack all known vectors for batch matrix multiplication
        # Shape: (N, 512)
        known_vectors = np.stack([known_face_encodings[n] for n in known_names], axis=0)
        
        for box in face_bboxes:
            # box is [x1, y1, x2, y2]
            face_tensor = face_utils.preprocess_face(rgb_frame, box)
            
            name = "Unknown"
            similarity_score = 0.0
            is_newly_marked = False
            
            if face_tensor is not None:
                # Get embedding for current face
                current_embedding = face_utils.get_embedding_from_face_tensor(face_tensor)
                
                # Compare with all known faces (Cosine Similarity)
                # Dot product of normalized vectors = Cosine Similarity
                similarities = np.dot(known_vectors, current_embedding)
                
                best_idx = int(np.argmax(similarities))
                best_similarity = float(similarities[best_idx])
                
                if best_similarity >= RECOGNITION_THRESHOLD:
                    name = known_names[best_idx]
                    similarity_score = best_similarity
                    
                    # Attendance Logic
                    current_time = time.time()
                    last_time = last_seen_timestamps.get(name)
                    
                    if (last_time is None) or (current_time - last_time > ATTENDANCE_COOLDOWN_SECONDS):
                        success, msg = mark_attendance(name)
                        if success:
                            last_seen_timestamps[name] = current_time
                            is_newly_marked = True
                        else:
                            attendance_error = msg
            
            matches_payload.append({
                "box": box, # [x1, y1, x2, y2]
                "name": name,
                "similarity": float(similarity_score),
                "newly_marked": is_newly_marked
            })
    else:
        # No known encodings, but we detected faces
        for box in face_bboxes:
            matches_payload.append({
                "box": box,
                "name": "Unknown (Empty DB)",
                "similarity": 0.0,
                "newly_marked": False
            })

    return jsonify({
        "success": True, 
        "matches": matches_payload, 
        "attendance_error": attendance_error
    })

if __name__ == "__main__":
    # Ensure encodings directory exists
    os.makedirs("encodings", exist_ok=True)
    # Run the app
    # Debug=True is great for development
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
