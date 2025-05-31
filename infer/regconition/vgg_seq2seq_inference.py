import numpy as np
np.bool = bool
np.int = int
np.float = float

import importlib, PIL, PIL._util
importlib.reload(PIL)
importlib.reload(PIL._util)

import os
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from tqdm import tqdm
import torch
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv(".env")

Image.ANTIALIAS = Image.LANCZOS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Cfg.load_config_from_name('vgg_seq2seq')
config['weights'] = os.getenv("MODEL_REGCONITION_PATH") 
config['device'] = device

predictor = Predictor(config)

def predict_vietocr(image):
    return predictor.predict(image)

def run_inference_on_folder(image_dir, output_file='data/text_recognition_output.txt'):

    predictions = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            pred_text = predict_vietocr(image)
            predictions.append(pred_text)
        except Exception as e:
            predictions.append(f"ERROR: {str(e)}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in predictions:
            f.write(line + '\n')

    print(f"Predictions written to {output_file}")

run_inference_on_folder('data/crops')
###