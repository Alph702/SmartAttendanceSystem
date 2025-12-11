# recognize.py
import cv2
import csv
import datetime
import time
import numpy as np
from utils import utils

ATTENDANCE_CSV = "attendance.csv"
THRESHOLD = 0.5  # start here and tune: lower = stricter

def mark_attendance(name):
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.datetime.utcnow().isoformat()])
    print(f"Marked {name} at {datetime.datetime.utcnow().isoformat()}")

def recognize(utils: utils, vc: cv2.VideoCapture):
    known = utils.load_all_encodings()
    if not known:
        print("No registered encodings found. Run register.py first.")
        return
    names = list(known.keys())
    vectors = np.stack([known[n] for n in names], axis=0)  # (N, D)

    last_seen = {}  # throttle per name (seconds)
    VC = vc
    print("Starting recognition. Press q to quit.")
    while True:
        ret, frame = VC.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = utils.detect_faces_rgb(rgb)
        for box in bboxes:
            x1,y1,x2,y2 = box
            face_tensor = utils.preprocess_face(rgb, box)
            if face_tensor is None:
                continue
            emb = utils.get_embedding_from_face_tensor(face_tensor)  # normalized
            # cosine similarity (emb · known) since normalized -> cos = dot
            sims = np.dot(vectors, emb)  # (N,)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= THRESHOLD:
                name = names[best_idx]
                now = time.time()
                if (name not in last_seen) or (now - last_seen[name] > 60):
                    mark_attendance(name)
                    last_seen[name] = now
                label = f"{name} {best_sim:.2f}"
                color = (0,255,0)
            else:
                label = f"Unknown {best_sim:.2f}"
                color = (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imshow("Mediapipe+ONNX Attendance - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize(utils())
