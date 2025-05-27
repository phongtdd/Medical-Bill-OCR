import os
import cv2
import numpy as np

def save_outputs(img, mask, extracted, output_dir, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    if mask.max() <= 1.0:
        mask = (mask > threshold).astype(np.uint8) * 255
    
    cv2.imwrite(os.path.join(output_dir, f"original.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"mask.jpg"), mask)
    cv2.imwrite(os.path.join(output_dir, f"extracted.jpg"), cv2.cvtColor(extracted, cv2.COLOR_RGB2BGR))
    print("Document Extracted")