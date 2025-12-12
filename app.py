import cv2
import threading
import time
import datetime
import csv
import base64
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, url_for
from utils import utils

app = Flask(__name__)

# --- Global State & Configuration ---
_utils = utils()  # Initialize utilities (loads ONNX models)

# Recognition State
known_encodings = {}
last_seen = {}  # {name: timestamp}
ATTENDANCE_CSV = "attendance.csv"
RECOGNITION_THRESHOLD = 0.5
ATTENDANCE_COOLDOWN = 60  # seconds

def load_encodings():
    global known_encodings
    try:
        known_encodings = _utils.load_all_encodings()
        print(f"Loaded {len(known_encodings)} encodings.")
    except Exception as e:
        print(f"Error loading encodings: {e}")
        known_encodings = {}

# Initial load
load_encodings()

# --- Helper Functions ---

def mark_attendance(name):
    """Marks attendance in CSV with timestamp. Returns (success, message)."""
    try:
        # Check if we can write to file
        with open(ATTENDANCE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, datetime.datetime.utcnow().isoformat()])
        print(f"Marked attendance for {name}")
        return True, "Success"
    except PermissionError:
        print(f"Permission Error marking attendance for {name}")
        return False, "Permission requested failed. Close Excel?"
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False, str(e)

def decode_base64_image(data_url):
    """
    Decodes a base64 data URL into an OpenCV image (BGR).
    """
    try:
        # data:image/jpeg;base64,.....
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/recognize')
def recognize_page():
    # Reload encodings ensure we have latest
    load_encodings()
    return render_template('recognize.html')

@app.route('/api/register_capture', methods=['POST'])
def register_capture():
    data = request.json
    name = data.get('name')
    image_data = data.get('image')
    
    if not name or not image_data:
        return jsonify({"success": False, "message": "Name and Image are required."})

    frame = decode_base64_image(image_data)
    if frame is None:
        return jsonify({"success": False, "message": "Invalid image data."})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = _utils.detect_faces_rgb(rgb)
    
    if not bboxes:
        return jsonify({"success": False, "message": "No face detected. Please try again."})

    # Get largest face
    bboxes_sorted = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    box = bboxes_sorted[0]
    
    face_tensor = _utils.preprocess_face(rgb, box)
    if face_tensor is None:
        return jsonify({"success": False, "message": "Face alignment failed."})

    emb = _utils.get_embedding_from_face_tensor(face_tensor)
    _utils.save_encoding(name, emb)
    
    # Reload encodings immediately
    load_encodings()
    
    return jsonify({"success": True, "message": f"Successfully registered {name}!"})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"success": False, "matches": []})

    frame = decode_base64_image(image_data)
    if frame is None:
        return jsonify({"success": False, "matches": []})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = _utils.detect_faces_rgb(rgb)
    
    matches_payload = []
    attendance_error = None
    
    names_list = list(known_encodings.keys())
    
    if names_list:
        vectors = np.stack([known_encodings[n] for n in names_list], axis=0)
        
        for box in bboxes:
            # box is [x1, y1, x2, y2]
            face_tensor = _utils.preprocess_face(rgb, box)
            
            name = "Unknown"
            sim = 0.0
            newly_marked = False
            
            if face_tensor is not None:
                emb = _utils.get_embedding_from_face_tensor(face_tensor)
                sims = np.dot(vectors, emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                
                if best_sim >= RECOGNITION_THRESHOLD:
                    name = names_list[best_idx]
                    sim = best_sim
                    
                    # Mark attendance logic
                    now = time.time()
                    if (name not in last_seen) or (now - last_seen[name] > ATTENDANCE_COOLDOWN):
                        success, msg = mark_attendance(name)
                        if success:
                            last_seen[name] = now
                            newly_marked = True
                        else:
                            attendance_error = msg
            
            matches_payload.append({
                "box": box, # [x1, y1, x2, y2]
                "name": name,
                "similarity": float(sim),
                "newly_marked": newly_marked
            })
    else:
        # No known encodings, but we detected faces
        for box in bboxes:
            matches_payload.append({
                "box": box,
                "name": "Unknown (No DB)",
                "similarity": 0.0
            })

    return jsonify({"success": True, "matches": matches_payload, "attendance_error": attendance_error})

if __name__ == "__main__":
    # Run the app
    # host='0.0.0.0' makes it accessible on network, use with caution
    app.run(host='0.0.0.0', debug=True, port=5000, threaded=True, use_reloader=True)
