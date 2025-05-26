import os
import re
import warnings

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv(".env")

tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_LABEL_PATH_1"))
model = AutoModelForSequenceClassification.from_pretrained(
    os.getenv("MODEL_LABEL_PATH_1")
).to(device)


def predict(text) -> int:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits[0], dim=-1).item()

    return predicted_class


def specific_rule(text: str) -> str:
    normalized_text = text.strip().replace(",", ".")

    # Quantity: starts with "SL:"
    if normalized_text.startswith("SL:"):
        return "quantity"

    # Data patterns
    if re.match(r"^Ngày\s*\d+", normalized_text, re.IGNORECASE):
        return "date"

    # Usage patterns
    usage_patterns = [
        r"\b(Sáng|Chiều|Tối|TRƯA)\b",
        r"\bUống\b",
        r"\b\d+\s*Viên\b",
    ]
    for pattern in usage_patterns:
        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            return "usage"

    # Drug  patterns
    drug_patterns = [
        r"^\d+\)",
        r"\b\d+(\.\d+)?\s*(mg|g|mcg|ml)\b",
        r"\b\d+(\.\d+)?(mg|g|mcg|ml)(\+\d+(\.\d+)?(mg|g|mcg|ml))*\b",
    ]
    for pattern in drug_patterns:
        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            return "drugname"

    return "other"


def predict_label(text: str) -> str:
    label = specific_rule(text)
    if label != "other":
        return label
    else:
        i: int = predict(text)
        if i == 1:
            return "diagnose"
        else:
            return "other"


if __name__ == "__main__":
    text = "Trong toàn phần (máu), Định lượng Creatinin (máu), Định lượng Glucose (Máu), Định lượng HbA Ic (Máu),"
    print(predict_label(text))
