import cv2
import threading
import time
import datetime
import csv
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from utils import utils

app = Flask(__name__)

# --- Global State & Configuration ---
camera_lock = threading.Lock()
camera = cv2.VideoCapture(0)
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
    """Marks attendance in CSV with timestamp."""
    try:
        with open(ATTENDANCE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, datetime.datetime.utcnow().isoformat()])
        print(f"Marked attendance for {name}")
    except Exception as e:
        print(f"Error marking attendance: {e}")

def process_frame(frame, mode='raw'):
    """
    Processes a frame based on the mode.
    Returns the processed frame (annotated).
    """
    if mode == 'raw':
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = _utils.detect_faces_rgb(rgb)

    if mode == 'register':
        # Just draw rectangles for feedback
        for (x1,y1,x2,y2) in bboxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    
    elif mode == 'recognize':
        # Recognition logic
        names_list = list(known_encodings.keys())
        if not names_list:
            # No database
            for (x1,y1,x2,y2) in bboxes:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, "No DB", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        else:
            vectors = np.stack([known_encodings[n] for n in names_list], axis=0)
            
            for box in bboxes:
                x1, y1, x2, y2 = box
                face_tensor = _utils.preprocess_face(rgb, box)
                
                label = "Unknown"
                color = (0, 0, 255) # Red for unknown
                
                if face_tensor is not None:
                    emb = _utils.get_embedding_from_face_tensor(face_tensor)
                    
                    # Cosine similarity
                    sims = np.dot(vectors, emb)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    
                    if best_sim >= RECOGNITION_THRESHOLD:
                        name = names_list[best_idx]
                        label = f"{name} ({best_sim:.2f})"
                        color = (0, 255, 0) # Green for known
                        
                        # Mark attendance logic
                        now = time.time()
                        if (name not in last_seen) or (now - last_seen[name] > ATTENDANCE_COOLDOWN):
                            mark_attendance(name)
                            last_seen[name] = now
                    else:
                        label = f"Unknown ({best_sim:.2f})"

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return frame

def generate_frames(mode='raw'):
    """Generator function for video streaming."""
    global camera
    while True:
        with camera_lock:
            if not camera.isOpened():
                camera = cv2.VideoCapture(0)
                time.sleep(0.1)
                
            success, frame = camera.read()
        
        if not success:
            time.sleep(0.1)
            continue

        # Process frame based on mode
        frame = process_frame(frame, mode=mode)

        # Encode header
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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

@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'raw')
    return Response(generate_frames(mode=mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/register_capture', methods=['POST'])
def register_capture():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({"success": False, "message": "Name is required."})

    # Capture one frame
    with camera_lock:
        if not camera.isOpened():
             return jsonify({"success": False, "message": "Camera not active."})
        success, frame = camera.read()
    
    if not success:
        return jsonify({"success": False, "message": "Failed to capture frame."})

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

if __name__ == "__main__":
    # Run the app
    # host='0.0.0.0' makes it accessible on network, use with caution
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
