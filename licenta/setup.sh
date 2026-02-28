#!/bin/bash

# Exit on error
set -e

echo "Starting OCR Environment Setup..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. System Dependencies
echo "--- Checking System Dependencies ---"

if command_exists apt-get; then
    echo "Detected apt-based system. Installing system packages..."
    
    # Update package list
    # sudo apt-get update

    # Install Tesseract and Romanian language pack
    if ! command_exists tesseract; then
        echo "Installing Tesseract OCR..."
        sudo apt-get install -y tesseract-ocr
    else
        echo "Tesseract OCR is already installed."
    fi

    # Check for Romanian language data
    if [ ! -f /usr/share/tesseract-ocr/4.00/tessdata/ron.traineddata ] && [ ! -f /usr/share/tesseract-ocr/5/tessdata/ron.traineddata ]; then
        echo "Installing Romanian language pack for Tesseract..."
        sudo apt-get install -y tesseract-ocr-ron
    else
        echo "Romanian language pack seems to be present."
    fi

    # Install libgl1 for OpenCV
    echo "Installing libgl1 (for OpenCV)..."
    sudo apt-get install -y libgl1

    # Install Python venv and pip if missing
    echo "Installing python3-venv and python3-pip..."
    sudo apt-get install -y python3-venv python3-pip git

else
    echo "Warning: 'apt-get' not found. Please manually ensure Tesseract (with Romanian support) and Python 3 are installed."
fi

# 2. Python Environment
echo "--- Setting up Python Environment ---"

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
else
    echo "Creating virtual environment '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python libraries..."
pip install easyocr pytesseract surya-ocr jupyter opencv-python torch torchvision matplotlib Pillow rapidfuzz jiwer pandas transformers accelerate addict easydict

echo "--- Setup Complete ---"
echo "To start the notebook, run:"
echo "source .venv/bin/activate"
echo "jupyter notebook ocr_comparison.ipynb"
