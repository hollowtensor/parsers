# RunPod Deployment — DeepSeek-OCR-2

Run DeepSeek-OCR-2 as an OpenAI-compatible API on a RunPod GPU Pod.

## Requirements

| Component | Tested With |
|-----------|-------------|
| GPU | RTX 4090 24 GB (any ≥20 GB VRAM works) |
| CUDA | 12.4 (also works with 11.8) |
| Python | 3.11 |
| vLLM | 0.8.5 |
| Model VRAM | ~6.3 GB (rest goes to KV cache) |

## Quick Start

### 1. Create a RunPod GPU Pod

- Go to [runpod.io](https://runpod.io) → Pods → Deploy
- Template: **RunPod PyTorch** (default)
- GPU: **RTX 4090** or **A40 48GB** or **A100**
- Disk: **50 GB+** (model weights ~7 GB + deps)
- Expose **TCP port 8000**

### 2. SSH In and Run Setup

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

cd /workspace
git clone https://github.com/hollowtensor/parsers.git
bash parsers/deploy/runpod/setup.sh
```

The setup script:
1. Clones `deepseek-ai/DeepSeek-OCR-2`
2. Upgrades PyTorch to 2.6.0 (auto-detects CUDA 11.8 vs 12.4)
3. Installs vLLM 0.8.5 (large download)
4. Pins `transformers==4.46.3` + `tokenizers==0.20.3` (version conflict warnings are expected and harmless — confirmed by the official repo)
5. Installs `flash-attn==2.7.3`, `addict`, `easydict`
6. Copies `server.py` into the vLLM directory
7. Pre-downloads the tokenizer

### 3. Start the Server

```bash
cd /workspace/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm
python server.py --port 8000 --gpu-mem 0.85
```

First start downloads model weights (~7 GB), then:
```
Loading model deepseek-ai/DeepSeek-OCR-2 ...
Model loading took 6.3336 GiB and 6.65 seconds
Server ready on 0.0.0.0:8000
```

To run in background:
```bash
nohup python server.py --port 8000 --gpu-mem 0.85 > /workspace/server.log 2>&1 &
tail -f /workspace/server.log  # watch logs
```

### 4. Connect from Your Machine

The RunPod proxy URL is: `https://<pod-id>-8000.proxy.runpod.net`

```bash
# Health check
curl https://<pod-id>-8000.proxy.runpod.net/health
```

```python
from parsers.methods.vlm.deepseek import DeepSeekOCRParser

parser = DeepSeekOCRParser(
    base_url="https://<pod-id>-8000.proxy.runpod.net/v1"
)

# Parse an image
doc = parser.parse("scan.png")

# Parse a PDF (pages sent concurrently with async)
doc = parser.parse("invoice.pdf")
print(doc.text)

# Access individual blocks with bounding boxes
for block in doc.pages[0].blocks:
    print(f"[{block.block_type}] {block.text[:80]}")
    if block.bbox:
        print(f"  bbox: ({block.bbox.x0}, {block.bbox.y0}) → ({block.bbox.x1}, {block.bbox.y1})")
```

## Server CLI Options

```
python server.py [OPTIONS]

  --host              Bind host           (default: 0.0.0.0)
  --port              Bind port           (default: 8000)
  --model             HF model ID         (default: deepseek-ai/DeepSeek-OCR-2)
  --gpu-mem           GPU memory fraction  (default: 0.85)
  --max-model-len     Max sequence length  (default: 8192)
  --max-concurrency   Max concurrent seqs  (default: 64)
  --tensor-parallel   TP GPU count         (default: 1)
```

## API Format

The server implements the OpenAI Chat Completions API subset:

```bash
curl -X POST https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
        {"type": "text", "text": "<|grounding|>Convert the document to markdown."}
      ]
    }],
    "max_tokens": 8192,
    "temperature": 0.0
  }'
```

## Prompt Modes

| Mode | Prompt Text | Grounding | Description |
|------|------------|-----------|-------------|
| document | `<\|grounding\|>Convert the document to markdown.` | Yes | Layout-aware markdown with bounding boxes |
| ocr | `Free OCR.` | No | Plain text extraction |
| locate | `Locate <\|ref\|>query<\|/ref\|> in the image.` | Yes | Find specific text/elements |
| describe | `Describe this image in detail.` | No | General image description |

## Troubleshooting

**pip timeout during vLLM install**: vLLM is large. Retry with `pip install --default-timeout=300 vllm==0.8.5`.

**Version conflict warnings**: `vllm 0.8.5 requires transformers>=4.51.1, but you have transformers 4.46.3` — this is expected and harmless. The official DeepSeek-OCR-2 README confirms these coexist fine.

**OOM on startup**: Lower `--gpu-mem` (e.g., `0.7`) or `--max-concurrency` (e.g., `16`).

**Port not reachable**: Make sure TCP port 8000 is exposed in the RunPod pod settings.
