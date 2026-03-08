import os
import sys

YOLOV9_DIR = os.path.join(os.path.dirname(__file__), "yolov9")
if YOLOV9_DIR not in sys.path:
    sys.path.insert(0, YOLOV9_DIR)

from models.experimental import attempt_load
from utils.torch_utils import select_device

m = attempt_load("yolov9/runs/train/fire_person/weights/best.pt", device=select_device("cpu"))
print("Fire model names:", m.names)