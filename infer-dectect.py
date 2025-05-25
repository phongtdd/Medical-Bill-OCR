import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLO Inference Script")
parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
args = parser.parse_args()

model = YOLO('path/to/your/model.pt', task='detect')

results = model(args.image_path)

results.show()
