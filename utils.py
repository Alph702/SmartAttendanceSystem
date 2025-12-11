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
from typing import List, Optional, Tuple, Any

class utils:
    def __init__(self, EMB_MODEL_PATH: str = "models/arcface_mobile.onnx", ENC_DIR: str = "encodings") -> None:
        self.EMB_MODEL_PATH = EMB_MODEL_PATH
        self.ENC_DIR = ENC_DIR
        self.mp_face_mesh = mp.solutions.face_mesh
        self.FACE_MESH = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                 max_num_faces=4,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        if not os.path.exists(EMB_MODEL_PATH):
            raise FileNotFoundError(f"Place embedding ONNX at: {EMB_MODEL_PATH}")
        self.emb_sess = ort.InferenceSession(EMB_MODEL_PATH, providers=["CPUExecutionProvider"])
        self._emb_input_name = self.emb_sess.get_inputs()[0].name

    def detect_with_mediapipe(self, img_bgr: np.ndarray) -> List[Any]:
        """
        Fast Mediapipe detection fallback. Returns boxes in pixel coords.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.FACE_MESH.process(img_rgb)
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
    
    def get_face_landmarks_mediapipe(self, img_bgr: np.ndarray, box: Tuple[int,int,int,int]) -> Optional[List[Tuple[int, int]]]:
        """
        Use Mediapipe Face Mesh to get landmarks for alignment.
        Returns the landmarks list (x,y) in pixel coords for the face closest to box.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.FACE_MESH.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        h, w = img_bgr.shape[:2]
        bx_cx = (box[0] + box[2]) / 2.0
        bx_cy = (box[1] + box[3]) / 2.0
        # choose the face whose bbox center is nearest to provided box center
        best = None
        best_dist = 1e9
        for lm in results.multi_face_landmarks:
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            cx = np.mean(xs)
            cy = np.mean(ys)
            dist = (cx - bx_cx)**2 + (cy - bx_cy)**2
            if dist < best_dist:
                best_dist = dist
                best = lm
        if best is None:
            return None
        # return a list of (x,y) for the landmarks
        lms = [(int(p.x * w), int(p.y * h)) for p in best.landmark]
        return lms
    
    def align_face_by_eyes(self, img_rgb: np.ndarray, landmarks: List[Tuple[int,int]], output_size=112, margin=0.25):
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

    def preprocess_for_model(self, face_rgb: np.ndarray, size=112):
        """
        face_rgb: uint8 HxWx3 RGB crop already aligned/resized
        returns: float32 NCHW tensor normalized to [0,1]
        """
        img = cv2.resize(face_rgb, (size, size)).astype(np.float32) / 255.0
        tensor = np.transpose(img, (2,0,1))[None, ...].astype(np.float32)
        return tensor

    def get_embedding_from_face_tensor(self, tensor: np.ndarray):
        out = self.emb_sess.run(None, {self._emb_input_name: tensor})
        emb = out[0].squeeze()
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)
    
    def detect_faces_rgb(self, img_rgb: np.ndarray):
        """
        Detects faces from an RGB image using mediapipe.
        """
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return self.detect_with_mediapipe(img_bgr)
    
    def preprocess_face(self, img_rgb: np.ndarray, box: Tuple[int,int,int,int], size=112):
        """
        Full pipeline from RGB image and a bounding box to a preprocessed face tensor.
        """
        # Mediapipe needs BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        landmarks = self.get_face_landmarks_mediapipe(img_bgr, box)
        if landmarks is None:
            return None  # Or handle error appropriately
        
        aligned_rgb = self.align_face_by_eyes(img_rgb, landmarks, output_size=size)
        
        tensor = self.preprocess_for_model(aligned_rgb, size=size)
        return tensor
    
    def save_encoding(self, name: str, emb: np.ndarray) -> None:
        path = os.path.join(self.ENC_DIR, f"{name}.npy")
        np.save(path, emb)
        print("Saved encoding:", path)

    def load_all_encodings(self):
        enc = {}
        for fn in os.listdir(self.ENC_DIR):
            if fn.endswith(".npy"):
                name = os.path.splitext(fn)[0]
                emb = np.load(os.path.join(self.ENC_DIR, fn))
                enc[name] = emb
        return enc

    def _resize_max_side(self, img, max_side=1280):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        return img, scale