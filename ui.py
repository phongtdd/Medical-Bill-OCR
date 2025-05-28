import importlib
import os
from typing import Any

import cv2
import numpy as np
import PIL
import PIL._util
import streamlit as st
from PIL import Image

importlib.reload(PIL)
importlib.reload(PIL._util)
import torch
from dotenv import load_dotenv

from backend.config import *
from backend.inference.detection.infer import seperate_text
from backend.inference.doc_segment.infer import infer_image
from backend.inference.recognition.infer_crnn import *
from backend.inference.recognition.infer_vietocr import *
from backend.inference.text_label.get_label import predict_label
from backend.inference.text_label.reformat_label import reformat_dict
from mongodb.update_db import *
from utils.load_model import *

load_dotenv(".env")

import io

from bson.binary import Binary


def decode_image(binary_data):
    image_bytes = io.BytesIO(binary_data)
    pil_image = Image.open(image_bytes)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def crop_receipt_image(img, model, device, save_mode=False, output_dir=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    extracted_doc, _ = infer_image(
        img=img, model=model, device=device, save_mode=save_mode, output_dir=output_dir
    )
    return extracted_doc


def separate_text_to_image(model, img, output_dir):
    detection_model = model
    seperate_text(detection_model, img, output_dir)


def extract_information(text, model, tokenizer) -> dict[str, list[str]]:
    labels = {}
    for line in text.split("\n"):
        label = predict_label(line, model, tokenizer)
        if label not in labels:
            labels[label] = []
        labels[label].append(line)
    return labels


def save_information(extracted_data, file_name) -> str:
    id = add_bill_dict(extracted_data, "CV_test_label")
    st.info(f"Saving {file_name}")
    return id


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
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_dir, filename)
                recognized_text = crnn_recognize_text(model, image_path, device)
                recognized_results.append(recognized_text)
        all_text = "\n".join(recognized_results)

    if model_number == 1:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_dir, filename)
                recognized_text = vietocr_recognize_text(model, image_path)
                recognized_results.append(recognized_text)
        all_text = "\n".join(recognized_results)

    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, "recognized_text.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(all_text)

    return all_text


# Main
# ----------------------------------------------
if __name__ == "__main__":
    segmentation_model = load_segmentation()
    detection_model = load_detection_model()
    label_model, tokenizer = load_label_model()

    st.title("Receipt Information Extraction")

    # Model selection
    model_name = st.selectbox(
        "Choose Text Recognition Model", ("CRNN", "VietOCR"), index=1
    )
    model_recognize_number = 0 if model_name == "CRNN" else 1
    regconition_model = (
        load_crnn_model() if model_recognize_number == 0 else load_vietocr()
    )

    # Input choice
    source_choice = st.radio(
        "Choose Image Source:", ["Upload New", "Use from Database"]
    )
    img_array = None
    filename = None

    if source_choice == "Upload New":
        uploaded_file = st.file_uploader(
            "Upload Receipt Image", type=["png", "jpg", "jpeg"]
        )
        custom_filename = st.text_input("Enter custom file name (without extension):")
        if uploaded_file:
            filename = (
                custom_filename.strip() + ".jpg"
                if custom_filename.strip()
                else uploaded_file.name
            )
            image = Image.open(uploaded_file)
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.image(image, caption="Uploaded Receipt")

    elif source_choice == "Use from Database":
        db_filename = st.text_input(
            "Enter existing file name in DB (e.g., VAIPE_P_TRAIN_1084.png)"
        )
        db_name = st.text_input(
            "Enter database name (default: CV_image)", "CV_train_image"
        )
        if st.button("Load Image from DB"):
            try:
                binary_data = get_image(db_filename, db_name)
                img_array = decode_image(binary_data)
                filename = db_filename
                st.image(img_array, channels="BGR", caption="Loaded from Database")
            except ValueError as e:
                st.error(str(e))

    # Proceed if image is ready
    if img_array is not None:
        # Crop
        if "cropped_image" in st.session_state:
            st.image(st.session_state["cropped_image"], caption="Cropped Receipt")

        # Crop button
        if st.button("Crop Receipt"):
            clear_output_dir(os.getenv("folder_segmentation"))
            cropped_image = crop_receipt_image(
                img_array,
                segmentation_model,
                device,
                save_mode=True,
                output_dir=os.getenv("folder_segmentation"),
            )
            st.image(cropped_image, caption="Cropped Receipt")
            st.session_state["cropped_image"] = cropped_image

        # Extract information button
        if "cropped_image" in st.session_state and st.button("Extract Information"):
            # Detect
            clear_output_dir(os.getenv("folder_detection"))
            separate_text_to_image(
                detection_model,
                st.session_state["cropped_image"],
                os.getenv("folder_detection"),
            )

            # Recognize
            clear_output_dir(os.getenv("folder_recognition"))
            recognized_text = recognize_all_detected_images(
                model_recognize_number,
                regconition_model,
                device,
                os.getenv("folder_detection"),
                os.getenv("folder_recognition"),
            )

            extracted_data = extract_information(
                recognized_text, label_model, tokenizer
            )
            extracted_data = extracted_data | {"name": [filename]}
            st.session_state["extracted_data"] = extracted_data

            # Display extracted information
            st.subheader("Extracted Information")
            st.json(extracted_data)

        if "extracted_data" in st.session_state and st.button("Reformat Information"):
            st.subheader("Reformatted Information")
            reformatted_data = reformat_dict(st.session_state["extracted_data"])
            st.json(reformatted_data)
            st.session_state["reformatted_data"] = reformatted_data

        # Store information button
        if "extracted_data" in st.session_state and st.button("Store Information"):
            if "reformatted_data" in st.session_state:
                file_id: str = save_information(
                    st.session_state["reformatted_data"], filename
                )
            else:
                file_id: str = save_information(
                    st.session_state["extracted_data"], filename
                )
            st.success(f"Information stored in MongoDB")
            st.session_state["file_id"] = file_id

    # Retrieve stored information from MongoDB
    st.subheader("Retrieve Stored Information")
    file_1_name = st.text_input("Enter file name to fetch from MongoDB:")
    db = st.text_input("Enter database name to fetch from MongoDB:")

    if st.button("Get Information"):
        try:
            bill = get_bill(file_1_name, db)
            st.subheader("Fetched Information")
            st.json(bill)

            # ➕ Add any additional processing here if needed
            # For example, if you want to use `bill` for downstream logic:
            st.success("Fetched information processed successfully.")
            # st.session_state["fetched_bill"] = bill  # If needed for reuse
        except ValueError as e:
            st.error(str(e))
