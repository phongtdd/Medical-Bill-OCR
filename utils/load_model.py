import importlib
import logging

import numpy as np
import PIL
import PIL._util
import segmentation_models_pytorch as smp
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

from backend.inference.recognition.crnn import CRNN

importlib.reload(PIL)
importlib.reload(PIL._util)

import os

import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from utils.utli import *

load_dotenv(".env")

Image.ANTIALIAS = Image.LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vietocr():
    config = Cfg.load_config_from_name("vgg_transformer")
    config["weights"] = os.getenv("MODEL_REGCONITION_PATH")
    config["device"] = device

    predictor = Predictor(config)
    return predictor


def load_viet_seq():
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["weights"] = os.getenv("MODEL_VGG_SEQ2_PATH")
    config["device"] = device

    predictor = Predictor(config)
    return predictor


def load_segmentation():
    model_path = os.getenv("MODEL_SEGMENTATION_PATH")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded model weights from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        raise

    return model


def load_crnn_model():
    crnn_model = CRNN(imgH=32, nc=1, nclass=len(full_alphabet) + 1, nh=256).to(device)
    crnn_model.load_state_dict(torch.load(os.getenv("MODEL_CRNN_PATH")))
    return crnn_model


def load_detection_model():
    detection_model = YOLO(os.getenv("MODEL_DETECT_PATH"))
    return detection_model


def load_label_model():
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_LABEL_PATH_V2"))
    model = AutoModelForSequenceClassification.from_pretrained(
        os.getenv("MODEL_LABEL_PATH_V2")
    ).to(device)
    return model, tokenizer
