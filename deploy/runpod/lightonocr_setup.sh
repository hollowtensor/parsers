#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# LightOnOCR-2-1B  ·  RunPod GPU Pod Setup
# ──────────────────────────────────────────────────────────────────────
#
# LightOnOCR-2 is natively supported by vLLM — no custom model code
# or wrapper server needed.  Just `vllm serve`.
#
# Tested environment:
#   RunPod template   : RunPod PyTorch (default)
#   GPU               : RTX 4090 24 GB  (any ≥8 GB VRAM works — model ~1.9 GB)
#   CUDA              : 12.4
#   Python            : 3.11
#   vLLM              : 0.15.0
#   transformers      : 5.0.0
#   Disk              : 40 GB+ recommended (vLLM + PyTorch + model weights)
#
# Usage:
#   ssh into your pod, then:
#     cd /workspace
#     git clone https://github.com/hollowtensor/parsers.git
#     bash parsers/deploy/runpod/lightonocr_setup.sh
#
# After setup, start the server:
#     vllm serve lightonai/LightOnOCR-2-1B \
#       --host 0.0.0.0 --port 8000 \
#       --limit-mm-per-prompt '{"image": 1}' \
#       --mm-processor-cache-gb 0 \
#       --no-enable-prefix-caching \
#       --gpu-memory-utilization 0.85
#
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "==> [1/4] Upgrading pip ..."
pip install --default-timeout=300 --upgrade pip

echo "==> [2/4] Installing vLLM 0.15.0 ..."
pip install --default-timeout=300 vllm==0.15.0

echo "==> [3/4] Installing transformers 5.0.0 (required by LightOnOCR-2) ..."
# vLLM pins transformers<5, but LightOnOCR-2 needs v5.
# This override works fine with vLLM 0.15.0 at runtime.
pip install --default-timeout=300 "transformers>=5.0.0" pypdfium2

echo "==> [4/4] Pre-downloading model weights ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lightonai/LightOnOCR-2-1B')
print('Model downloaded.')
"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Start the server:"
echo "    vllm serve lightonai/LightOnOCR-2-1B \\"
echo "      --host 0.0.0.0 --port 8000 \\"
echo "      --limit-mm-per-prompt '{\"image\": 1}' \\"
echo "      --mm-processor-cache-gb 0 \\"
echo "      --no-enable-prefix-caching \\"
echo "      --gpu-memory-utilization 0.85"
echo ""
echo "  Background:"
echo "    nohup vllm serve lightonai/LightOnOCR-2-1B \\"
echo "      --host 0.0.0.0 --port 8000 \\"
echo "      --limit-mm-per-prompt '{\"image\": 1}' \\"
echo "      --mm-processor-cache-gb 0 \\"
echo "      --no-enable-prefix-caching \\"
echo "      --gpu-memory-utilization 0.85 > /workspace/server.log 2>&1 &"
echo ""
echo "  Expose port 8000 in RunPod dashboard, then connect:"
echo "    https://<pod-id>-8000.proxy.runpod.net/health"
echo "════════════════════════════════════════════════════════════════"
