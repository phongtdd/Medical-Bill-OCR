from ultralytics import YOLO
import sys
import os

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = 'data/images/train/VAIPE_P_TRAIN_0.png'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Image not found: {file_path}")

model = YOLO('kqua/runs/detect/train/weights/detectext/best.pt')

results = model(file_path, device='0')

results[0].show()