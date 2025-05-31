import torch

from backend.inference.recognition.crnn import *
from backend.inference.recognition.utlis import *

device = torch.device("cuda")


def crnn_recognize_text(model, image_path, device):
    with torch.no_grad():
        image = process_image(image_path)
        image = image.unsqueeze(0).to(device)
        output = model(image)
        decoded_texts = beam_search_decode(output, beam_width=10, blank=0)
        cleaned_text = clean_decoded_text(decoded_texts)
        cleaned_text = cleaned_text.strip()
    return cleaned_text
