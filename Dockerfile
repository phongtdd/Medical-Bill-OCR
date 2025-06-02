# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Use a base image with Python
ARG PYTHON_VERSION=3.10.16
FROM python:${PYTHON_VERSION}-slim AS base

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory inside the container
WORKDIR /MEDICAL_BILL_ORC

# Copy code and dependencies into the image
COPY requirements.txt .
RUN pip install --default-timeout=1800 -r requirements.txt
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Set the command to run the application
CMD ["streamlit", "run", "ui.py"]
