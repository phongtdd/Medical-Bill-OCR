from typing import Any

import pandas as pd
from pandas.core.frame import DataFrame

from mongodb.update_db import *
from mongodb.util import *
from predict_text_label.reformat_label import reformat_dict

if __name__ == "__main__":
    data_folder = "data"
    file_name = "test1.txt"
    try:
        id = add_bill_txt(data_folder, file_name)
        print(f"Added bill with ID: {id}")
    except Exception as e:
        print(f"Error adding bill {file_name}: {e}")
    # try:
    #     d = dict_from_txt(f"{data_folder}/{file_name}")
    #     print(d)
    #     print(reformat_dict(d))
    # except Exception as e:
    #     print(f"Error processing file {file_name}: {e}")
