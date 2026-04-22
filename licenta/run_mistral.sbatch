#!/bin/bash
#SBATCH --job-name=mistral_ocr
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=mistral_%j.out
#SBATCH --error=mistral_%j.err

# Activăm mediul virtual.
source .venv/bin/activate

# Pentru Ollama, setăm calea unde va salva modelul ca să nu umple `/tmp` aiurea
export OLLAMA_MODELS=$HOME/.ollama/models

# Pornim serverul de Ollama pe nodul curent, în background
ollama serve & 
OLLAMA_PID=$!

echo "Asteptăm 10s să pornească Ollama server..."
sleep 10

# Rulăm scriptul de postprocesare OCR
python mistrall_ocr_postprocess.py

# Oprim procesul de Ollama
kill $OLLAMA_PID
