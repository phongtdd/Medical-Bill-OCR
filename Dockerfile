# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Use a base image with Python
ARG PYTHON_VERSION=3.10.16
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory inside the container
WORKDIR /MEDICAL_BILL_ORC

# Copy code and dependencies into the image
COPY requirements.txt .
RUN pip install --default-timeout=1800 -r requirements.txt
COPY . .

# Set the command to run the application
CMD ["/bin/bash"]

