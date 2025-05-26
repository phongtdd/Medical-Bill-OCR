from typing import Any

import pandas as pd
from dotenv import load_dotenv

from predict_text_label.get_label import predict_label
from predict_text_label.reformat_label import reformat_dict


def dict_from_json(file_name: str) -> dict[str, Any]:
    df = pd.read_json(file_name)
    d = {}
    for _, row in df.iterrows():
        text: str = row["text"]
        label: str = row["label"]
        if label not in d:
            d[label] = []
        d[label].append(text)
    return d


def dict_from_txt(file_name: str) -> dict[str, Any]:
    with open(file_name, "r") as f:
        lines: list[str] = f.readlines()
    d = {}
    for line in lines:
        label = predict_label(line)
        if label not in d:
            d[label] = []
        d[label].append(line.strip())
    return reformat_dict(d)


if __name__ == "__main__":
    load_dotenv()
    print(
        dict_from_json(
            "/home/sag/Working/Hust/Medical-Bill-OCR/data/vaipe-p/public_train/label/VAIPE_P_TRAIN_0.json"
        )
    )
