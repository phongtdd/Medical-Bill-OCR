from typing import Any

import pandas as pd
from pandas.core.frame import DataFrame

from get_label import predict_label
from mongodb.update_db import add_bill


def save_db(file, data_name) -> str:
    with open(file, "r", encoding="utf-8") as f:
        lines: list[str] = f.readlines()
    d = {}
    for line in lines:
        label: str = predict_label(line)
        if label not in d:
            d[label] = []
        d[label].append(line.strip())
    id: str = add_bill(d, data_name)
    return id


if __name__ == "__main__":
    file = "data/test.txt"
    data_name = "CV"
    id = save_db(file, data_name)
    print(f"Data saved with ID: {id}")
