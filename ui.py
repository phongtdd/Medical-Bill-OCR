import importlib
import io
import os
from typing import Any

import cv2
import numpy as np
import PIL
import PIL._util
import streamlit as st
from bson.binary import Binary
from PIL import Image

# Import your existing modules (assuming they exist)
try:
    importlib.reload(PIL)
    importlib.reload(PIL._util)
    import torch
    from dotenv import load_dotenv

    from backend.config import *
    from backend.inference.detection.infer import seperate_text
    from backend.inference.doc_segment.infer import infer_image
    from backend.inference.recognition.infer_crnn import *
    from backend.inference.recognition.infer_vietocr import *
    from backend.inference.recognition.vgg_seq2seq_inference import (
        vgg_seq_recognize_text,
    )
    from backend.inference.text_label.get_label import predict_label
    from backend.inference.text_label.reformat_label import reformat_dict
    from mongodb.update_db import *
    from utils.load_model import *

    load_dotenv(".env")
except ImportError:
    st.error(
        "Some backend modules are not available. Please ensure all dependencies are installed."
    )


# Custom CSS for modern dashboard look
def apply_custom_css():
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .dashboard-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #4a5568;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-description {
        color: #a0aec0;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .feature-tag {
        display: inline-block;
        background: #2d3748;
        color: #68d391;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem 0.5rem 0.25rem 0;
    }
    
    .launch-button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .launch-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(72, 187, 120, 0.4);
    }
    
    .intro-text {
        color: #a0aec0;
        text-align: center;
        margin-bottom: 3rem;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .footer {
        text-align: center;
        color: #718096;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #2d3748;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Dark theme */
    .stApp {
        background-color: #1a202c;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Your existing utility functions
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
        line = line.strip()
        if not line:
            continue
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


def get_recognizer(model_number: int, model, device):
    if model_number == 0:
        return lambda image_path: crnn_recognize_text(model, image_path, device)
    elif model_number == 1:
        return lambda image_path: vietocr_recognize_text(model, image_path)
    elif model_number == 2:
        return lambda image_path: vgg_seq_recognize_text(model, image_path)
    else:
        raise ValueError(f"Unsupported model number: {model_number}")


def recognize_all_detected_images(
    model_number: int, model, device, input_dir: str, output_dir: str
) -> str:
    recognizer = get_recognizer(model_number, model, device)

    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    recognized_results = [
        recognizer(os.path.join(input_dir, filename)) for filename in image_files
    ]

    all_text = "\n".join(recognized_results)

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "recognized_text.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(all_text)

    return all_text


# Extract Information Page
def extract_information_page():
    st.markdown(
        '<div class="main-header">🧾 Extract Receipt Information</div>',
        unsafe_allow_html=True,
    )

    # Model selection
    col1, col2 = st.columns([1, 1])
    with col1:
        model_name = st.selectbox(
            "Choose Text Recognition Model",
            ("CRNN", "VietOCR", "VGG_Seq2Seq"),
            index=1,
            key="extract_model",
        )

    with col2:
        source_choice = st.radio(
            "Choose Image Source:",
            ["Upload New", "Use from Database"],
            key="extract_source",
        )

    # Initialize session state for extract page
    if "extract_img_array" not in st.session_state:
        st.session_state["extract_img_array"] = None
    if "extract_filename" not in st.session_state:
        st.session_state["extract_filename"] = None

    # Image loading section
    if source_choice == "Upload New":
        uploaded_file = st.file_uploader(
            "Upload Receipt Image", type=["png", "jpg", "jpeg"], key="extract_uploader"
        )
        if uploaded_file:
            filename = uploaded_file.name
            custom_name = st.text_input(
                "Enter custom name for the image (optional)",
                value=filename,
                key="extract_custom_name",
            )
            if custom_name:
                filename = custom_name

            image = Image.open(uploaded_file)
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            st.session_state["extract_img_array"] = img_array
            st.session_state["extract_filename"] = filename

            st.image(image, caption="Uploaded Receipt", use_column_width=True)

    elif source_choice == "Use from Database":
        col1, col2 = st.columns([2, 1])
        with col1:
            db_filename = st.text_input(
                "Enter existing file name in DB", key="extract_db_filename"
            )
        with col2:
            db_name = st.text_input(
                "Database name", "CV_train_image", key="extract_db_name"
            )

        if st.button("Load Image from DB", key="extract_load_db"):
            try:
                binary_data = get_item(db_filename, db_name)
                img_array = decode_image(binary_data)
                filename = db_filename

                st.session_state["extract_img_array"] = img_array
                st.session_state["extract_filename"] = filename

                st.image(img_array, channels="BGR", caption="Loaded from Database")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    # Processing section
    img_array = st.session_state.get("extract_img_array", None)
    filename = st.session_state.get("extract_filename", None)

    if img_array is not None:
        st.markdown("---")
        st.subheader("🔄 Processing Steps")

        # Step 1: Crop Receipt
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "📋 Crop Receipt", key="extract_crop", use_container_width=True
            ):
                with st.spinner("Cropping receipt..."):
                    try:
                        segmentation_model = load_segmentation()
                        clear_output_dir(os.getenv("folder_segmentation"))
                        cropped_image = crop_receipt_image(
                            img_array,
                            segmentation_model,
                            device,
                            save_mode=True,
                            output_dir=os.getenv("folder_segmentation"),
                        )
                        st.session_state["extract_cropped_image"] = cropped_image
                        st.success("✅ Receipt cropped successfully!")
                    except Exception as e:
                        st.error(f"Error cropping receipt: {str(e)}")

        # Display cropped image if available
        if "extract_cropped_image" in st.session_state:
            st.image(
                st.session_state["extract_cropped_image"],
                caption="Cropped Receipt",
                use_column_width=True,
            )

            # Step 2: Extract Information
            with col2:
                if st.button(
                    "🔍 Extract Information",
                    key="extract_info",
                    use_container_width=True,
                ):
                    with st.spinner("Extracting information..."):
                        try:
                            # Text detection
                            detection_model = load_detection_model()
                            clear_output_dir(os.getenv("folder_detection"))
                            separate_text_to_image(
                                detection_model,
                                st.session_state["extract_cropped_image"],
                                os.getenv("folder_detection"),
                            )

                            # Text recognition
                            model_recognize_number = (
                                0
                                if model_name == "CRNN"
                                else 1
                                if model_name == "VietOCR"
                                else 2
                            )
                            recognition_model = (
                                load_crnn_model()
                                if model_recognize_number == 0
                                else load_vietocr()
                                if model_recognize_number == 1
                                else load_viet_seq()
                            )

                            clear_output_dir(os.getenv("folder_recognition"))
                            recognized_text = recognize_all_detected_images(
                                model_recognize_number,
                                recognition_model,
                                device,
                                os.getenv("folder_detection"),
                                os.getenv("folder_recognition"),
                            )

                            # Label extraction
                            label_model, tokenizer = load_label_model()
                            extracted_data = extract_information(
                                recognized_text, label_model, tokenizer
                            )
                            extracted_data = extracted_data | {"name": [filename]}

                            st.session_state["extract_extracted_data"] = extracted_data
                            st.success("✅ Information extracted successfully!")

                        except Exception as e:
                            st.error(f"Error extracting information: {str(e)}")

        # Results section
        if "extract_extracted_data" in st.session_state:
            st.markdown("---")
            st.subheader("📊 Extracted Information")
            st.json(st.session_state["extract_extracted_data"])

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button(
                    "🔧 Reformat Information",
                    key="extract_reformat",
                    use_container_width=True,
                ):
                    with st.spinner("Reformatting..."):
                        try:
                            reformatted_data = reformat_dict(
                                st.session_state["extract_extracted_data"]
                            )
                            st.session_state["extract_reformatted_data"] = (
                                reformatted_data
                            )
                            st.success("✅ Information reformatted!")
                            st.json(reformatted_data)
                        except Exception as e:
                            st.error(f"Error reformatting: {str(e)}")

            with col2:
                if st.button(
                    "💾 Store Information",
                    key="extract_store",
                    use_container_width=True,
                ):
                    with st.spinner("Storing information..."):
                        try:
                            data_to_store = (
                                st.session_state.get("extract_reformatted_data", None)
                                or st.session_state["extract_extracted_data"]
                            )
                            file_id = save_information(data_to_store, filename)
                            st.session_state["extract_file_id"] = file_id
                            st.success(f"✅ Information stored successfully!")
                        except Exception as e:
                            st.error(f"Error storing information: {str(e)}")


# Retrieve Information Page
def retrieve_information_page():
    st.markdown(
        '<div class="main-header">🔍 Retrieve Stored Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        file_name = st.text_input(
            "Enter file name to retrieve:",
            placeholder="e.g., VAIPE_P_TRAIN_1084.png",
            key="retrieve_filename",
        )
    with col2:
        db_name = st.text_input(
            "Database name:", "CV_test_label", key="retrieve_db_name"
        )

    if st.button("🔍 Get Information", key="retrieve_get", use_container_width=True):
        if file_name and db_name:
            with st.spinner("Retrieving information..."):
                try:
                    item = get_item(file_name, db_name)

                    if isinstance(item, bytes):
                        # It's an image
                        img_array = decode_image(item)
                        st.image(
                            img_array,
                            channels="BGR",
                            caption="Retrieved Image",
                            use_column_width=True,
                        )

                    elif isinstance(item, dict):
                        # It's structured data
                        st.subheader("📋 Retrieved Information")
                        st.json(item)

                        # Add download functionality
                        if st.button("📥 Download as JSON", key="retrieve_download"):
                            import json

                            json_str = json.dumps(item, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"{file_name}_data.json",
                                mime="application/json",
                            )
                    else:
                        st.warning("⚠️ Unsupported data type returned from database.")

                except Exception as e:
                    st.error(f"❌ Error retrieving information: {str(e)}")
        else:
            st.warning("⚠️ Please enter both file name and database name.")


# Main Dashboard
def main_dashboard():
    st.markdown(
        '<div class="main-header">Receipt Information Processing Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="intro-text">
    Welcome to the Receipt Information Processing Dashboard, which provides advanced tools for extracting 
    information from receipt images using OCR and machine learning models, and retrieving stored data.
    Select one of the processing tools below to begin your analysis.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">
                🧾 Extract Information
            </div>
            <div class="card-description">
                Process receipt images with advanced OCR techniques. Extract text information, 
                analyze content structure, and store results with multiple recognition models and preprocessing options.
            </div>
            <div>
                <span class="feature-tag">Image Segmentation</span>
                <span class="feature-tag">OCR Recognition</span>
                <span class="feature-tag">Text Classification</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Launch Extract Tool", key="launch_extract", use_container_width=True
        ):
            st.session_state.page = "extract"
            st.rerun()

    with col2:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">
                🔍 Retrieve Information
            </div>
            <div class="card-description">
                Access and retrieve previously processed receipt data from the database. 
                Search stored information, view extracted content, and download results in various formats.
            </div>
            <div>
                <span class="feature-tag">Database Search</span>
                <span class="feature-tag">Data Export</span>
                <span class="feature-tag">Content Viewing</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Launch Retrieve Tool", key="launch_retrieve", use_container_width=True
        ):
            st.session_state.page = "retrieve"
            st.rerun()

    st.markdown(
        """
    <div class="footer">
        Receipt Processing Dashboard © 2024 | Built with Streamlit
    </div>
    """,
        unsafe_allow_html=True,
    )


# Main App
def main():
    st.set_page_config(
        page_title="Receipt Processing Dashboard",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_custom_css()

    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    # Navigation
    if st.session_state.page != "dashboard":
        if st.button("← Back to Dashboard", key="back_button"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.markdown("---")

    # Page routing
    if st.session_state.page == "dashboard":
        main_dashboard()
    elif st.session_state.page == "extract":
        extract_information_page()
    elif st.session_state.page == "retrieve":
        retrieve_information_page()


if __name__ == "__main__":
    main()
