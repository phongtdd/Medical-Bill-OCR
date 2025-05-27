import cv2
import torch
import os
from backend.inference.doc_segment.process import extract_document
from backend.inference.doc_segment.save_output import save_outputs

IMG_SIZE = (512, 512)

def infer_image(img, model, device, save_mode=True, output_dir="output"):
    img_resized = cv2.resize(img, IMG_SIZE)
    
    extracted_doc, pred_mask = extract_document(img, model, device, image_size=IMG_SIZE)
    
    if save_mode:
        save_outputs(img_resized, pred_mask, extracted_doc, output_dir=output_dir)
    
    return extracted_doc, pred_mask