from typing import Any

import pandas as pd
from pandas.core.frame import DataFrame

from backend.inference.text_label.get_label import predict_label
from mongodb.update_db import *
from mongodb.util import *

if __name__ == "__main__":
    file_name = "VAIPE_P_TRAIN_1084.png"
    db_name = "CV_train_image"
    id = get_image(file_name, db_name)
    print(id)
