import os
import cv2
from PIL import Image
import numpy as np

DOC_IMG_PATH = r"vaipe-p\public_test\image"
DOC_MSK_PATH = r"vaipe-p\public_test\masks"

'Create a mask full of white for vaipe-p dataset'

os.makedirs(DOC_MSK_PATH, exist_ok=True)

for img_name in os.listdir(DOC_IMG_PATH):
    img_path = os.path.join(DOC_IMG_PATH, img_name)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    mask = np.ones((H, W, 3), dtype=np.uint8) * 255
    mask_path = os.path.join(DOC_MSK_PATH, img_name)
    cv2.imwrite(mask_path, mask)