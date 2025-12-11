from utils import utils
from register import register_student
from recognize import recognize

if __name__ == "__main__":
    _utils = utils()
    while True:
        options: int = int(input("===== Smart Attendance System =====\n1. Regester\n2. Recignize\n0. Quit\n>"))
        if options == 1:
            name: str = input("Enter student name (no spaces recommended): ").strip()
            register_student(name, _utils)
        if options == 2:
            recognize(_utils)
        if options == 0:
            break
