from typing import Any

import pandas as pd
from pandas.core.frame import DataFrame

from mongodb.update_db import *
from mongodb.util import *

if __name__ == "__main__":
    data_folder = "data"
    file_name = "test.txt"
    try:
        id = add_bill_txt(data_folder, file_name)
        print(f"Added bill with ID: {id}")
    except Exception as e:
        print(f"Error adding bill {file_name}: {e}")
