import cv2
import os

def seperate_text(model, image, output_dir):
    results = model(image,device ='cpu')[0]
    
    os.makedirs(output_dir, exist_ok=True)

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        cropped = image[y1:y2, x1:x2]

        filename = f"{output_dir}/object_{i}_class{class_id}_conf{conf:.2f}.png"
        cv2.imwrite(filename, cropped)