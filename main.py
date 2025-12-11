from utils import utils
from register import register_student
from recognize import recognize
from cv2 import VideoCapture

if __name__ == "__main__":
    _utils = utils()
    _vc = VideoCapture(0)
    while True:
        options: int = int(input("===== Smart Attendance System =====\n1. Regester\n2. Recignize\n0. Quit\n>"))
        if options == 1:
            name: str = input("Enter student name (no spaces recommended): ").strip()
            register_student(name, _utils, _vc)
        if options == 2:
            recognize(_utils, _vc)
        if options == 0:
            break
