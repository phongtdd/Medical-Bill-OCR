import streamlit as st
import json
import os
from datetime import datetime
from PIL import Image
import io
import cv2
from inference.doc_segment.infer import infer_image
from inference.doc_segment.model import load_model
from inference.detection.infer import seperate_text
from inference.recognition.infer_crnn import *
import numpy as np
from config import *
from ultralytics import YOLO

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
       
def extract_information(image):
    
    return {
        "store_name": "Sample Store",
        "date": "2025-05-26",
        "total": 29.99,
        "items": [
            {"name": "Item 1", "price": 9.99},
            {"name": "Item 2", "price": 20.00}
        ]
    }

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

def recognize_all_detected_images(model, device, input_dir, output_dir):
    recognized_results = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            recognized_text = recognition_text(model, image_path, device)

            # if isinstance(recognized_text, bytes):
            #     recognized_text = recognized_text.decode('utf-8', errors='ignore')

            recognized_results.append(recognized_text)

    all_text = '\n'.join(recognized_results)
    print(all_text)

    output_filepath = os.path.join(output_dir, "recognized_text.txt")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(all_text)

    return output_filepath



#Main
#----------------------------------------------
if __name__ == "__main__":
    segmentation_model, device = load_segmentation_model(MODEL_PATH['segmentation'])
    detection_model = load_detection_model(MODEL_PATH['detection'])
    crnn_model = load_crnn_model(MODEL_PATH['recognition_crnn'])
    
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
            clear_output_dir(OUTPUT_PATH['detection'])
            separate_text_to_image(MODEL_PATH['detection'], st.session_state['cropped_image'], OUTPUT_PATH['detection'])
            clear_output_dir(OUTPUT_PATH['recognition'])
            recognition_text_path = recognize_all_detected_images(crnn_model, device, OUTPUT_PATH['detection'], OUTPUT_PATH['recognition'])
            
            extracted_data = extract_information(st.session_state['cropped_image'])
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