import streamlit as st
import json
import os
import numpy as np
from datetime import datetime
from PIL import Image
import io
import cv2
from ultralytics import YOLO
import importlib, PIL, PIL._util
importlib.reload(PIL)
importlib.reload(PIL._util)
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import torch
from torchvision import transforms
from dotenv import load_dotenv

from backend.inference.doc_segment.infer import infer_image
from backend.inference.doc_segment.model import load_model
from backend.inference.detection.infer import seperate_text
from backend.inference.recognition.infer_crnn import *
from backend.inference.recognition.infer_vietocr import *
from backend.config import *
from predict_text_label.get_label import *

def load_vietocr_model(model_path):
    load_dotenv(".env")
    Image.ANTIALIAS = Image.LANCZOS
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = model_path
    config['device'] = 'cuda'
    predictor = Predictor(config)
    return predictor

def load_crnn_model(model_path):
    crnn_model = CRNN(imgH=32, nc=1, nclass=len(full_alphabet) + 1, nh=256).to(device)
    crnn_model.load_state_dict(torch.load(model_path))
    return crnn_model
 
def load_segmentation_model(model_path):
    segmentation_model, device = load_model(model_path)
    return segmentation_model, device

def load_detection_model(model_path):
    detection_model = YOLO(model_path)
    return detection_model

def crop_receipt_image(img, model, device, save_mode=False, output_dir=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    extracted_doc, _ = infer_image(img=img, model=model, device=device, save_mode=save_mode, output_dir=output_dir)
    return extracted_doc

def separate_text_to_image(model_pth, img, output_dir):
    detection_model = load_detection_model(model_pth)
    seperate_text(detection_model, img, output_dir)
       
def extract_information(text):
    labels = {}
    for line in text.split('\n'):
        label_of_text = predict_label(line)
        labels[label_of_text] = line
    return labels

def save_information(extracted_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"receipt_data_{timestamp}.json"
    history_file = "receipt_history.json"
    
    with open(file_path, 'w') as f:
        json.dump(extracted_data, f, indent=4)
    
    # Update history
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    
    history.append({
        "timestamp": timestamp,
        "file_path": file_path,
        "data": extracted_data
    })
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    return file_path

def clear_output_dir(output_dir):
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def recognize_all_detected_images(model_number, model, device, input_dir, output_dir):
    recognized_results = []
    if model_number == 0:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                recognized_text = crnn_recognize_text(model, image_path, device)

                recognized_results.append(recognized_text)

        all_text = '\n'.join(recognized_results)

    if model_number == 1:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                recognized_text = vietocr_recognize_text(model, image_path)
                recognized_results.append(recognized_text)

        all_text = '\n'.join(recognized_results)
    output_filepath = os.path.join(output_dir, "recognized_text.txt")
    output_filepath = output_filepath.replace('\\', '/')
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(all_text)

    return all_text



#Main
#----------------------------------------------
if __name__ == "__main__":
    segmentation_model, device = load_segmentation_model(MODEL_PATH['segmentation'])
    detection_model = load_detection_model(MODEL_PATH['detection'])
    crnn_model = load_crnn_model(MODEL_PATH['recognition_crnn'])
    vietocr_model = load_vietocr_model(MODEL_PATH['recognition_vietocr'])
    model_recognize_number = 1
    if model_recognize_number == 0:
        regconition_model = crnn_model
    if model_recognize_number == 1:
        regconition_model = vietocr_model

    st.title("Receipt Information Extraction")

    uploaded_file = st.file_uploader("Upload Receipt Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(image, caption="Uploaded Receipt")
        
        # Crop
        # Show the cropped image if already stored in session
        if 'cropped_image' in st.session_state:
            st.image(st.session_state['cropped_image'], caption="Cropped Receipt")

        # Crop button
        if st.button("Crop Receipt"):
            clear_output_dir(OUTPUT_PATH['segmentation'])
            cropped_image = crop_receipt_image(img_array, segmentation_model, device, save_mode=True, output_dir=OUTPUT_PATH['segmentation'])
            st.image(cropped_image, caption="Cropped Receipt")
            st.session_state['cropped_image'] = cropped_image

        # Extract information button
        if 'cropped_image' in st.session_state and st.button("Extract Information"):
            # Detect
            clear_output_dir(OUTPUT_PATH['detection'])
            separate_text_to_image(MODEL_PATH['detection'], st.session_state['cropped_image'], OUTPUT_PATH['detection'])

            # Recognize
            clear_output_dir(OUTPUT_PATH['recognition'])
            recognized_text = recognize_all_detected_images(1,regconition_model, device, OUTPUT_PATH['detection'], OUTPUT_PATH['recognition'])
            
            # Extract information
            extracted_data = extract_information(recognized_text)
            st.session_state['extracted_data'] = extracted_data

            # Display extracted information
            st.subheader("Extracted Information")
            st.json(extracted_data)

            # Store information button
            if st.button("Store Information"):
                file_path = save_information(extracted_data)
                st.success(f"Information stored in {file_path}")

        
    # Display previous processing results
    st.subheader("Previous Processing Results")
    history_file = "receipt_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        for record in history:
            with st.expander(f"Receipt from {record['timestamp']}"):
                st.json(record['data'])
    else:
        st.write("No previous processing results found.")