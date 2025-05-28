import argparse
import os

import cv2
from ultralytics import YOLO


def enlarge_box(xyxy, scale=1.6, img_shape=None):
    x1, y1, x2, y2 = xyxy
    h = y2 - y1
    pad_w = h * scale

    new_x1 = max(x1 - pad_w, 0)
    new_x2 = min(x2 + pad_w, img_shape[1])
    new_y1 = int(y1)
    new_y2 = int(y2)

    return [int(new_x1), new_y1, int(new_x2), new_y2]


def run(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to read image: {img_path}")
        return

    output_folder = os.path.join("data", "crops")
    os.makedirs(output_folder, exist_ok=True)

    model = YOLO("/home/sag/Working/Hust/Medical-Bill-OCR/model/detection/best.pt")
    results = model(img, device="cuda", iou=0.4)
    result = results[0]

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = enlarge_box([x1, y1, x2, y2], scale=1.6, img_shape=img.shape)
        crop = img[y1:y2, x1:x2]
        save_path = os.path.join(output_folder, f"crop_{i + 1}.jpg")
        print(f"Saving to: {save_path}")
        cv2.imwrite(save_path, crop)

    print(f"✅ Saved {i + 1} crops to 'data/{output_folder}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    run(args.img)
