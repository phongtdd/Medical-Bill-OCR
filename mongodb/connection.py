import os

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv(".env")
uri = os.getenv("MONGODB_URI")


if not uri:
    raise ValueError("MONGODB_URI environment variable is not set.")
else:
    client = MongoClient(uri)

database = client["HUST"]
