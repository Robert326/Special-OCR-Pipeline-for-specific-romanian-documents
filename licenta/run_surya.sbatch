#!/bin/bash
#SBATCH --job-name=surya_ocr
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=surya_%j.out
#SBATCH --error=surya_%j.err

# Activăm mediul virtual.
source ../.venv/bin/activate

# Este o idee bună să setăm un director specific pentru cache-ul de modele Hugging Face 
# (folosit de Surya), pentru a nu umple memoria temporară (`/tmp`) a nodului de pe Grid.
export HF_HOME=$HOME/.cache/huggingface

# Traversăm în folderul OCR așa încât căile din script (ex: "../preprocessed_images") să meargă,
# presupunând că pornești `sbatch` din folderul principal.
cd OCR

# Rulăm scriptul Python
python ocr_compare.py
