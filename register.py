# register.py
import cv2
from utils import utils

def register_student(name, utils: utils, vc: cv2.VideoCapture):
    vc = vc
    print("Press SPACE when your face is visible to capture and register. Press q to quit.")
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = utils.detect_faces_rgb(rgb)
        for (x1,y1,x2,y2) in bboxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow("Register - Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # space pressed -> capture
            if not bboxes:
                print("No face detected. Try again.")
                continue
            # Choose largest bbox
            bboxes_sorted = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            box = bboxes_sorted[0]
            face_tensor = utils.preprocess_face(rgb, box)
            if face_tensor is None:
                print("Crop failed. Try again.")
                continue
            emb = utils.get_embedding_from_face_tensor(face_tensor)
            utils.save_encoding(name, emb)
            print(f"Registered {name} (embedding saved).")
            break
        if key == ord("q"):
            break
    # vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter student name (no spaces recommended): ").strip()
    register_student(name, utils())
