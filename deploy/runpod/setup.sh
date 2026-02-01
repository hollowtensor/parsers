#!/usr/bin/env bash
# Run this once on a fresh RunPod GPU Pod (A40 48GB+ recommended).
# Tested with RunPod PyTorch 2.6 + CUDA 11.8 template.

set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace}"
cd "$WORK_DIR"

echo "==> Cloning DeepSeek-OCR-2..."
if [ ! -d "DeepSeek-OCR-2" ]; then
    git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
fi

echo "==> Installing Python dependencies..."
pip install -U pip
pip install \
    vllm==0.8.5 \
    flash-attn==2.7.3 \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    PyMuPDF \
    img2pdf \
    einops \
    easydict \
    addict \
    Pillow \
    numpy \
    fastapi \
    uvicorn

echo "==> Downloading model weights (this takes a while on first run)..."
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True)
print('Tokenizer cached.')
# Model weights will be pulled by vLLM on first server start.
"

echo ""
echo "==> Setup complete. Start the server with:"
echo "    cd $WORK_DIR && python DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/server.py"
echo ""
echo "    Or copy server.py there first if not already done:"
echo "    cp deploy/runpod/server.py DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/"
