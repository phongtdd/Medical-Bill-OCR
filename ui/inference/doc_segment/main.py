import cv2
from infer import infer_image
from model import load_model

if __name__ == "__main__":
    img_path = r"TestImage\4_1.png"
    model_path=r"Model\best_segmentation_unet_resnet50_vaipep.pth"
    output_dir="Output"
    
    model, device = load_model(model_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    infer_image(img=img, model=model, device=device, save_mode=True, output_dir=output_dir)