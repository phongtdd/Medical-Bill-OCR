import os
from typing import Any

from bson.binary import Binary

from mongodb.connection import database
from mongodb.util import *


def add_bill_json(
    data_folder: str, file_name: str, db_name: str = "CV_train_label"
) -> str:
    file_path = f"{data_folder}/{file_name}"
    d: dict[str, list[str]] = dict_from_json(file_path)
    d = {"name": file_name, **d}
    collection = database[db_name]
    existing = collection.find_one(d)
    if existing:
        print("Duplicate document found")
        return existing["_id"]
    result = collection.insert_one(d)
    return result.inserted_id


def add_bill_txt(
    data_folder: str, file_name: str, db_name: str = "CV_test_label"
) -> str:
    file_path = f"{data_folder}/{file_name}"
    d: dict[str, list[str]] = dict_from_txt(file_path)
    d = {"name": file_name, **d}
    collection = database[db_name]
    existing = collection.find_one(d)
    if existing:
        print("Duplicate document found")
        return existing["_id"]
    result = collection.insert_one(d)
    return result.inserted_id


def get_bill(file_name: str, db_name: str = "CV_train_label") -> dict[str, Any]:
    collection = database[db_name]
    doc = collection.find_one({"name": file_name})
    if not doc:
        raise ValueError(f"No document found with this name: {file_name}")
    return doc


def add_image(data_folder: str, file_name: str, db_name: str = "CV_train_image") -> str:
    with open(f"{data_folder}/{file_name}", "rb") as f:
        image_data = f.read()
    doc = {
        "name": file_name,
        "image": Binary(image_data),
    }
    collection = database[db_name]
    existing = collection.find_one(doc)
    if existing:
        print("Duplicate document found")
        return existing["_id"]
    result = collection.insert_one(doc)
    return result.inserted_id


def get_image(file_name: str, name: str = "CV_image") -> None:
    collection = database[name]
    doc = collection.find_one({"name": file_name})
    if not doc:
        raise ValueError(f"No document found with this name: {file_name}")
    return doc["image"]


if __name__ == "__main__":
    data_folder = "data/vaipe-p/public_train/label"
    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".json"):
            continue
        try:
            id = add_bill_json(data_folder, file_name)
            print(f"Added bill with ID: {id}")
        except Exception as e:
            print(f"Error adding bill {file_name}: {e}")
