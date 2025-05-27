import torch
import segmentation_models_pytorch as smp
import logging
from torchinfo import summary

def load_model(model_path=r"Model\best_segmentation_unet_resnet50_vaipep.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded model weights from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        raise
    
    return model, device