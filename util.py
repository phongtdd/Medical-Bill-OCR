import json
import os
from datetime import datetime


def extract_information(text):
    labels = {}
    labels["quantity"] = []
    labels["date"] = []
    labels["usage"] = []
    labels["drugname"] = []
    labels["diagnose"] = []
    labels["other"] = []
    for line in text.split("\n"):
        label_of_text = predict_label(line)
        labels[label_of_text].append(line)
    return labels


def save_information(extracted_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"receipt_data_{timestamp}.json"
    history_file = "receipt_history.json"

    with open(file_path, "w") as f:
        json.dump(extracted_data, f, indent=4)

    # Update history
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

    history.append(
        {"timestamp": timestamp, "file_path": file_path, "data": extracted_data}
    )

    with open(history_file, "w") as f:
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
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, filename)
            recognized_text = recognition_text(model, image_path, device)

            # if isinstance(recognized_text, bytes):
            #     recognized_text = recognized_text.decode('utf-8', errors='ignore')

            recognized_results.append(recognized_text)

    all_text = "\n".join(recognized_results)
    print(all_text)

    output_filepath = os.path.join(output_dir, "recognized_text.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(all_text)

    return output_filepath
