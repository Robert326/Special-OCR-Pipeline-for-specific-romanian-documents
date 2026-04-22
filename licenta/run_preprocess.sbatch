#!/bin/bash
#SBATCH --job-name=prep_ocr
#SBATCH --partition=haswell
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err

# Activăm mediul virtual.
# Modifică calea dacă '.venv' este în altă parte.
source /home/robert/Special-OCR-Pipeline-for-specific-romanian-documents/licenta/.venv/bin/activate

cd Preprocess
python PrepareForOCR.py
