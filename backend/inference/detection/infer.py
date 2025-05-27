import cv2
import os

def enlarge_box(xyxy, scale=1.6, img_shape=None):
    x1, y1, x2, y2 = xyxy
    h = y2 - y1
    pad_w = h * scale

    new_x1 = max(x1 - pad_w, 0)
    new_x2 = min(x2 + pad_w, img_shape[1])
    new_y1 = int(y1)
    new_y2 = int(y2)

    return [int(new_x1), new_y1, int(new_x2), new_y2]

def seperate_text(model, image, output_folder):
    results = model(image, device='cpu', iou=0.4)
    result = results[0]
    os.makedirs(output_folder, exist_ok=True)

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = enlarge_box([x1, y1, x2, y2], scale=1.6, img_shape=image.shape)
        crop = image[y1:y2, x1:x2]
        save_path = os.path.join(output_folder, f"crop_{i+1}.jpg")
        cv2.imwrite(save_path, crop)