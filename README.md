# Medical-Bill-OCR
Mini-project for course of Intro to Computer Vision

# Set up
Install the required packages by using the following command:
```bash
pip install -r requirements.txt
```
# Text Detection
This module provides a simple interface to run object detection inference on an image .
Here is an example of using it:
```bash
python .\detection\infer-detection.py --img data/images/val/VAIPE_P_TRAIN_1103.png
```

# Download models
The trained models are available on the Hugging Face model hub: [Hugging Face model hub](https://huggingface.co/Sag1012/Medical_Bill_OCR)

You can see so many models here. However, if you want to try the best model of our experiments, you can try the model in the following folder:
- Predict label: [diagnose_predict_v2](https://huggingface.co/Sag1012/Medical_Bill_OCR/tree/main/diagnose_predict_v2)
