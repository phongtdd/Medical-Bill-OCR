from typing import Any

import pandas as pd
import torch
from pandas.core.frame import DataFrame

from backend.inference.text_label.get_label import predict_label
from mongodb.update_db import *
from mongodb.util import *
from utils.load_model import load_crnn_model

if __name__ == "__main__":
    crnn = load_crnn_model()
    print(crnn)
