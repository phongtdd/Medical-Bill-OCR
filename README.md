# Medical Bill OCR

A mini-project developed for the *Introduction to Computer Vision* course.

This project extracts and processes textual information from medical bills using Optical Character Recognition (OCR) techniques.

---

# Installation

## Install via pip
- Clone project
```bash
git clone https://github.com/phongtdd/Medical-Bill-OCR.git
cd Medical-Bill-OCR
```
- Install requirement
```bash
pip install -r requirements.txt
```

- Run application
```bash
python -m streamlit run ui.py
```

## Environment Configuration via Docker
For ease of use, we also provide a installation package via a docker image. You can set up Chronos's docker step-by-step as follow:

- Pull Chronos's docker image:
```bash
docker pull sagp1012/medical-bill-ocr:latest
```
- Run a docker container:
```bash
docker run --gpus all -p 8501:8501 sagp1012/medical-bill-ocr:latest
```

# Download models
The trained models are available on the Hugging Face model hub: [Hugging Face model hub](https://huggingface.co/Sag1012/Medical_Bill_OCR)
