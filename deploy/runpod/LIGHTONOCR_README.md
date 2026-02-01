# RunPod Deployment — LightOnOCR-2-1B

Run LightOnOCR-2-1B as an OpenAI-compatible API on a RunPod GPU Pod using vLLM.

LightOnOCR-2 is natively supported by vLLM — no custom server code needed.

## Requirements

| Component | Tested With |
|-----------|-------------|
| GPU | RTX 4090 24 GB (any >=8 GB VRAM works) |
| CUDA | 12.4 |
| Python | 3.11 |
| vLLM | 0.15.0 |
| transformers | 5.0.0 |
| Model VRAM | ~1.9 GB (rest goes to KV cache) |
| Disk | 40 GB+ (vLLM + PyTorch ~20 GB, model ~2 GB) |

## Quick Start

### 1. Create a RunPod GPU Pod

- Go to [runpod.io](https://runpod.io) -> Pods -> Deploy
- Template: **RunPod PyTorch** (default)
- GPU: **RTX 4090** or any GPU with >=8 GB VRAM
- Disk: **40 GB+**
- Expose **TCP port 8000**

### 2. SSH In and Run Setup

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

cd /workspace
git clone https://github.com/hollowtensor/parsers.git
bash parsers/deploy/runpod/lightonocr_setup.sh
```

The setup script:
1. Upgrades pip
2. Installs vLLM 0.15.0 (includes PyTorch 2.9, CUDA libs)
3. Installs transformers 5.0.0 (required by LightOnOCR-2)
4. Pre-downloads model weights (~2 GB)

### 3. Start the Server

```bash
vllm serve lightonai/LightOnOCR-2-1B \
  --host 0.0.0.0 --port 8000 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85
```

Startup output:
```
Resolved architecture: LightOnOCRForConditionalGeneration
Model loading took 1.88 GiB memory and 12.36 seconds
Application startup complete.
```

To run in background:
```bash
nohup vllm serve lightonai/LightOnOCR-2-1B \
  --host 0.0.0.0 --port 8000 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 > /workspace/server.log 2>&1 &

tail -f /workspace/server.log  # watch logs
```

### 4. Connect from Your Machine

The RunPod proxy URL is: `https://<pod-id>-8000.proxy.runpod.net`

```bash
# Health check
curl https://<pod-id>-8000.proxy.runpod.net/health

# List models
curl https://<pod-id>-8000.proxy.runpod.net/v1/models
```

#### Python example

```python
import base64, httpx

SERVER = "https://<pod-id>-8000.proxy.runpod.net/v1"

with open("document.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = httpx.post(
    f"{SERVER}/chat/completions",
    json={
        "model": "lightonai/LightOnOCR-2-1B",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        }],
        "max_tokens": 4096,
        "temperature": 0.2,
    },
    timeout=120.0,
)
print(resp.json()["choices"][0]["message"]["content"])
```

#### Gradio demo

```bash
# Install dependencies
pip install gradio httpx pypdfium2 pillow numpy

# Launch demo
python demo/lightonocr_app.py \
  --server-url https://<pod-id>-8000.proxy.runpod.net/v1
```

Open http://localhost:7861 in your browser.

## API Format

LightOnOCR uses the standard OpenAI Chat Completions API. No text prompt
is needed — just send an image in the user message:

```bash
curl -X POST https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lightonai/LightOnOCR-2-1B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}}
      ]
    }],
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9
  }'
```

## Model Variants

| Model | Has Bbox | Description |
|-------|----------|-------------|
| `lightonai/LightOnOCR-2-1B` | No | Best overall OCR (recommended) |
| `lightonai/LightOnOCR-2-1B-bbox` | Yes | Best bounding box detection |
| `lightonai/LightOnOCR-2-1B-base` | No | Base model, ideal for fine-tuning |
| `lightonai/LightOnOCR-2-1B-bbox-base` | Yes | Base bbox model |
| `lightonai/LightOnOCR-2-1B-ocr-soup` | No | Merged variant for robustness |
| `lightonai/LightOnOCR-2-1B-bbox-soup` | Yes | Merged OCR + bbox combined |

To serve a different variant, replace the model name in the `vllm serve` command.

## Bbox Output Format

Bbox model variants output image references with coordinates:

```
![image](image_1.png)100,200,500,600
```

Coordinates are normalized to 0-1000. To convert to pixel coordinates:
```python
px = coord * image_width / 1000
py = coord * image_height / 1000
```

## Server CLI Options

```
vllm serve lightonai/LightOnOCR-2-1B [OPTIONS]

  --host                   Bind host             (default: 0.0.0.0)
  --port                   Bind port             (default: 8000)
  --gpu-memory-utilization GPU memory fraction    (default: 0.9)
  --max-model-len          Max sequence length    (default: 16384)
  --tensor-parallel-size   TP GPU count           (default: 1)
  --limit-mm-per-prompt    Max images per request (set to '{"image": 1}')
  --mm-processor-cache-gb  MM processor cache     (set to 0)
  --no-enable-prefix-caching  Disable prefix caching
```

## Image Preprocessing Tips

From the official LightOnOCR docs:
- Render PDFs at **200 DPI** (scale factor 2.77)
- Target longest dimension: **1540px**
- Maintain aspect ratio to preserve text geometry

## Troubleshooting

**`all_special_tokens_extended` error**: vLLM pins transformers<5 but LightOnOCR needs v5. Make sure you install transformers 5.0.0 *after* vLLM. This override works at runtime with vLLM 0.15.0.

**Disk space during install**: vLLM 0.15.0 pulls PyTorch 2.9 (~900 MB) and CUDA libs. Use a pod with at least 40 GB disk. Clear pip cache if needed: `pip cache purge`.

**OOM on startup**: Lower `--gpu-memory-utilization` (e.g., `0.7`).

**Port not reachable**: Make sure TCP port 8000 is exposed in RunPod pod settings.

**Streaming not working**: vLLM 0.15.0 supports streaming natively via `"stream": true` in the API request.

## Comparison with DeepSeek-OCR-2

| Feature | LightOnOCR-2-1B | DeepSeek-OCR-2 |
|---------|-----------------|----------------|
| Model size | 1B (~1.9 GB VRAM) | ~6.3 GB VRAM |
| vLLM support | Native (`vllm serve`) | Custom server wrapper |
| Setup complexity | Simple (just vLLM) | Complex (custom model registration) |
| Output format | Clean HTML/Markdown | Grounding tags with bboxes |
| Bbox support | Separate `-bbox` variants | Built into grounding mode |
| Speed | ~5.7 pages/s on H100 | Varies |
