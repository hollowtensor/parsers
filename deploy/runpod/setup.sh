#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# DeepSeek-OCR-2  ·  RunPod GPU Pod Setup
# ──────────────────────────────────────────────────────────────────────
#
# Tested environment:
#   RunPod template   : RunPod PyTorch (default)
#   GPU               : RTX 4090 24 GB  (any ≥20 GB VRAM works)
#   CUDA              : 12.4   (also works with 11.8)
#   Python            : 3.11
#   Model VRAM usage  : ~6.3 GB  (rest goes to KV cache)
#
# What this script does (in order):
#   1. Clones DeepSeek-OCR-2 repo
#   2. Upgrades PyTorch to 2.6.0 (required by vLLM 0.8.5)
#   3. Installs vLLM 0.8.5  (large download — can take a few minutes)
#   4. Pins transformers==4.46.3 + tokenizers==0.20.3
#      (vLLM pulls newer versions, but DeepSeek-OCR-2 needs these;
#       the version conflict warnings are expected and harmless —
#       confirmed by the official repo README)
#   5. Installs flash-attn and remaining deps
#   6. Copies server.py into the vLLM directory
#   7. Pre-downloads the tokenizer
#
# Usage:
#   ssh into your pod, then:
#     git clone https://github.com/hollowtensor/parsers.git
#     bash parsers/deploy/runpod/setup.sh
#
# After setup, start the server:
#     cd /workspace/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm
#     python server.py --port 8000 --gpu-mem 0.85
#
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

# ── 1. Clone model repo ────────────────────────────────────────────────
echo "==> [1/7] Cloning DeepSeek-OCR-2..."
if [ ! -d "DeepSeek-OCR-2" ]; then
    git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
else
    echo "    Already cloned, skipping."
fi

# ── 2. Upgrade PyTorch ──────────────────────────────────────────────────
# vLLM 0.8.5 requires PyTorch 2.6.0.  RunPod default template ships 2.4.x.
# Detect CUDA version and pick the right index URL.
echo "==> [2/7] Upgrading PyTorch to 2.6.0..."
CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12")
if [ "$CUDA_MAJOR" = "11" ]; then
    PIP_INDEX="https://download.pytorch.org/whl/cu118"
else
    PIP_INDEX="https://download.pytorch.org/whl/cu124"
fi
pip install --default-timeout=300 torch==2.6.0 torchvision --index-url "$PIP_INDEX"

# ── 3. Install vLLM ────────────────────────────────────────────────────
# This is ~500 MB+ and brings in fastapi, uvicorn, einops, pillow, numpy, etc.
echo "==> [3/7] Installing vLLM 0.8.5 (large download)..."
pip install --default-timeout=300 vllm==0.8.5

# ── 4. Pin transformers + tokenizers ────────────────────────────────────
# vLLM installs transformers>=5.0 and tokenizers>=0.21, but DeepSeek-OCR-2's
# custom model code requires the older versions.  The pip warnings about
# version conflicts are harmless — vLLM + transformers coexist fine at runtime.
echo "==> [4/7] Pinning transformers==4.46.3 + tokenizers==0.20.3..."
pip install transformers==4.46.3 tokenizers==0.20.3

# ── 5. Install flash-attn + remaining deps ─────────────────────────────
echo "==> [5/7] Installing flash-attn and remaining dependencies..."
pip install --default-timeout=300 flash-attn==2.7.3 addict easydict

# ── 6. Copy server.py ──────────────────────────────────────────────────
VLLM_DIR="$WORK_DIR/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm"
echo "==> [6/7] Copying server.py into $VLLM_DIR ..."
cp "$SCRIPT_DIR/server.py" "$VLLM_DIR/server.py"

# ── 7. Pre-download tokenizer ──────────────────────────────────────────
echo "==> [7/7] Pre-downloading tokenizer (model weights download on first server start)..."
python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True)
print('    Tokenizer cached.')
"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Start the server:"
echo "    cd $VLLM_DIR"
echo "    python server.py --port 8000 --gpu-mem 0.85"
echo ""
echo "  Expose port 8000 in RunPod dashboard, then connect:"
echo "    https://<pod-id>-8000.proxy.runpod.net/health"
echo "════════════════════════════════════════════════════════════════"
