"""
utils.py
- SCRFD ONNX detection (fallback to Mediapipe detection if SCRFD not found)
- Mediapipe Face Mesh for landmarks-based alignment
- ONNX runtime for embeddings (Mobile-ArcFace / MobileFaceNet)
- simple save/load encodings (one .npy per user)
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import math
from typing import List, Tuple

# ---------- Paths ----------
SCRFD_PATH = "models/scrfd.onnx"         # put your scrfd tiny ONNX here
EMB_MODEL_PATH = "models/arcface_mobile.onnx"  # put your mobile arcface ONNX here
ENC_DIR = "encodings"
os.makedirs(ENC_DIR, exist_ok=True)

# ---------- Mediapipe face mesh (for alignment) ----------
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=False,
                                 max_num_faces=4,
                                 refine_landmarks=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

# ---------- Load ONNX embedding model ----------
if not os.path.exists(EMB_MODEL_PATH):
    raise FileNotFoundError(f"Place embedding ONNX at: {EMB_MODEL_PATH}")

emb_sess = ort.InferenceSession(EMB_MODEL_PATH, providers=["CPUExecutionProvider"])
_emb_input_name = emb_sess.get_inputs()[0].name
# Expect shape like [1,3,112,112] - adjust preprocess if different

# ---------- Optional SCRFD detector loader ----------
SCRFD_AVAILABLE = os.path.exists(SCRFD_PATH)
if SCRFD_AVAILABLE:
    scrfd_sess = ort.InferenceSession(SCRFD_PATH, providers=["CPUExecutionProvider"])
    scrfd_input_name = scrfd_sess.get_inputs()[0].name
    # SCRFD outputs depend on model; we assume it outputs [1, N, 5] or [1, N, 15] (bbox + score + kps)
else:
    scrfd_sess = None



# ---------- Helpers ----------
def resize_max_side(img, max_side=1280):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img, scale

def detect_with_scrfd(img_bgr: np.ndarray, conf_thresh=0.45):
    """
    Run a minimal SCRFD postprocess. This function assumes the SCRFD model returns
    bounding boxes and scores in a standard layout. Different SCRFD ONNX outputs vary,
    so this is a best-effort generic routine that may require small adjustments per model.
    Returns list of boxes (x1,y1,x2,y2) in original image coords.
    """
    if not SCRFD_AVAILABLE:
        return []

    img, scale = resize_max_side(img_bgr, max_side=1280)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(img_rgb, (640, 640))  # many scrfd variants expect 640
    inp = inp.astype(np.float32)
    inp = np.transpose(inp, (2, 0, 1))[None, ...]  # NCHW
    # run
    out = scrfd_sess.run(None, {scrfd_input_name: inp})
    # Heuristic parsing: many SCRFD variants return boxes as Nx5 or Nx15; try to find a 2D array
    boxes = []
    scores = []
    try:
        # flatten to array of floats
        arr = np.concatenate([np.ravel(o) for o in out])
        # try to parse outputs into Nx? rows using heuristics — fallback: skip
        # Many pre-exported SCRFDs include output[0] as [1,N,15] or [1,N,5]
        candidate = None
        for o in out:
            a = np.array(o)
            if a.ndim == 3 and a.shape[2] >= 5:
                candidate = a[0]
                break
            if a.ndim == 2 and a.shape[1] >= 5:
                candidate = a
                break
        if candidate is None:
            return []
        
        # candidate rows: [x1, y1, x2, y2, score, ...] or [x1,y1,w,h,score,...]
        raw_boxes = []
        raw_scores = []
        
        for row in candidate:
            score = float(row[4])
            if score < conf_thresh:
                continue
            
            # Heuristic: detect whether coords are center/wh or corners
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            
            # if row looks like center/w/h (normalized or not), we might need conversion
            # But standard SCRFD usually outputs [x1, y1, x2, y2, score] in pixel coords relative to input size (640x640)
            
            # Check if normalized (0..1)
            if x2 <= 1.0 and y2 <= 1.0:
                # normalized coords (0..1) -> convert to original image scale
                x1 = max(0, int(x1 * w)); y1 = max(0, int(y1 * h))
                x2 = min(w, int((x1 + x2 * w))); y2 = min(h, int((y1 + y2 * h)))
            else:
                # assume already in pixel coords relative to 640 input
                # Need to map back from resized input to original image size
                scale_x = w / 640.0
                scale_y = h / 640.0
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
            
            raw_boxes.append([x1, y1, x2 - x1, y2 - y1]) # NMSBoxes expects [x, y, w, h]
            raw_scores.append(score)

        if not raw_boxes:
            return []

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(raw_boxes, raw_scores, score_threshold=conf_thresh, nms_threshold=0.4)
        
        for i in indices:
            # cv2.dnn.NMSBoxes returns a list of indices, sometimes as [[i], [j]] or [i, j]
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            bx, by, bw, bh = raw_boxes[idx]
            boxes.append((bx, by, bx + bw, by + bh))

    except Exception as e:
        # If parser fails, return empty
        print("SCRFD parsing failed:", e)
        return []
    return boxes

def detect_with_mediapipe(img_bgr: np.ndarray):
    """
    Fast Mediapipe detection fallback. Returns boxes in pixel coords.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(img_rgb)
    h, w = img_bgr.shape[:2]
    boxes = []
    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]
            xmin = int(max(0, min(xs) * w))
            xmax = int(min(w, max(xs) * w))
            ymin = int(max(0, min(ys) * h))
            ymax = int(min(h, max(ys) * h))
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def get_face_landmarks_mediapipe(img_bgr: np.ndarray, box: Tuple[int,int,int,int]):
    """
    Use Mediapipe Face Mesh to get landmarks for alignment.
    Returns the landmarks list (x,y) in pixel coords for the face closest to box.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    bx_cx = (box[0] + box[2]) / 2.0
    bx_cy = (box[1] + box[3]) / 2.0
    # choose the face whose bbox center is nearest to provided box center
    best = None; best_dist = 1e9
    for lm in results.multi_face_landmarks:
        xs = [p.x * w for p in lm.landmark]
        ys = [p.y * h for p in lm.landmark]
        cx = np.mean(xs); cy = np.mean(ys)
        dist = (cx - bx_cx)**2 + (cy - bx_cy)**2
        if dist < best_dist:
            best_dist = dist
            best = lm
    if best is None:
        return None
    # return a list of (x,y) for the landmarks
    lms = [(int(p.x * w), int(p.y * h)) for p in best.landmark]
    return lms

def align_face_by_eyes(img_rgb: np.ndarray, landmarks: List[Tuple[int,int]], output_size=112, margin=0.25):
    """
    Simple similarity transform aligner using eye centers.
    landmarks: mediapipe 468 points; left eye approx indices and right eye indices are used.
    Returns a square aligned crop (RGB uint8) sized output_size.
    """
    # Mediapipe landmarks: left eye approx indexes (33..133 region) and right eye indexes (263..362)
    left_eye_idx = [33, 133, 160, 159, 158, 157, 173]
    right_eye_idx = [263, 362, 387, 386, 385, 384, 398]
    # compute eye centers
    lx = np.mean([landmarks[i][0] for i in left_eye_idx])
    ly = np.mean([landmarks[i][1] for i in left_eye_idx])
    rx = np.mean([landmarks[i][0] for i in right_eye_idx])
    ry = np.mean([landmarks[i][1] for i in right_eye_idx])
    # desired positions
    eye_mid = ((lx+rx)/2.0, (ly+ry)/2.0)
    # calculate angle
    dy = ry - ly
    dx = rx - lx
    angle = math.degrees(math.atan2(dy, dx))
    # scale: distance between eyes in pixels
    dist = math.hypot(dx, dy)
    # desired distance between eyes in output
    desired_eye_dist = output_size * 0.35
    scale = desired_eye_dist / (dist + 1e-6)
    # get rotation matrix around eye_mid
    M = cv2.getRotationMatrix2D(eye_mid, angle, scale)
    # translate so eye_mid maps to center
    t_x = output_size * 0.5
    t_y = output_size * 0.4
    M[0,2] += (t_x - eye_mid[0])
    M[1,2] += (t_y - eye_mid[1])
    aligned = cv2.warpAffine(img_rgb, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned

def preprocess_for_model(face_rgb: np.ndarray, size=112):
    """
    face_rgb: uint8 HxWx3 RGB crop already aligned/resized
    returns: float32 NCHW tensor normalized to [0,1]
    """
    img = cv2.resize(face_rgb, (size, size)).astype(np.float32) / 255.0
    # if model needs mean/std or BGR, change here
    tensor = np.transpose(img, (2,0,1))[None, ...].astype(np.float32)
    return tensor

def get_embedding_from_face_tensor(tensor: np.ndarray):
    out = emb_sess.run(None, {_emb_input_name: tensor})
    emb = out[0].squeeze()
    # Normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)

def detect_faces_rgb(img_rgb: np.ndarray):
    """
    Detects faces from an RGB image, choosing the best available detector.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Force MediaPipe for better reliability
    return detect_with_mediapipe(img_bgr)

def preprocess_face(img_rgb: np.ndarray, box: Tuple[int,int,int,int], size=112):
    """
    Full pipeline from RGB image and a bounding box to a preprocessed face tensor.
    """
    # Mediapipe needs BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    landmarks = get_face_landmarks_mediapipe(img_bgr, box)
    if landmarks is None:
        return None  # Or handle error appropriately
    
    aligned_rgb = align_face_by_eyes(img_rgb, landmarks, output_size=size)
    
    tensor = preprocess_for_model(aligned_rgb, size=size)
    return tensor

# ---------- simple encoding storage ----------
def save_encoding(name: str, emb: np.ndarray):
    path = os.path.join(ENC_DIR, f"{name}.npy")
    np.save(path, emb)
    print("Saved encoding:", path)

def load_all_encodings():
    enc = {}
    for fn in os.listdir(ENC_DIR):
        if fn.endswith(".npy"):
            name = os.path.splitext(fn)[0]
            emb = np.load(os.path.join(ENC_DIR, fn))
            enc[name] = emb
    return enc

