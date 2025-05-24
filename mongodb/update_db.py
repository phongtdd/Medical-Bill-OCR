from .connection import database


def add_bill(bill_dict: dict[str, str], name: str = "CV") -> str:
    collection = database[name]
    existing = collection.find_one(bill_dict)
    if existing:
        print("Duplicate document found")
        return existing["_id"]
    result = collection.insert_one(bill_dict)
    return result.inserted_id
