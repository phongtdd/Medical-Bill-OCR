import cv2
import os
from ultralytics import YOLO

# Load your trained model
model = YOLO("kqua/runs/detect/train/weights/detectext/best.pt")  # Replace with your trained model path

# Load the image
image_path = "data/images/train/VAIPE_P_TRAIN_0.png"  # Replace with your image path
image = cv2.imread(image_path)

# Run inference
results = model(image_path,device ='cpu')[0]

# Create output directory
output_dir = "detected_crops"
os.makedirs(output_dir, exist_ok=True)

# Loop through detections and save crops
for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    class_id = int(box.cls[0])
    conf = float(box.conf[0])
    cropped = image[y1:y2, x1:x2]

    filename = f"{output_dir}/object_{i}_class{class_id}_conf{conf:.2f}.png"
    cv2.imwrite(filename, cropped)

print("Cropping complete. Check the 'detected_crops' folder.")
