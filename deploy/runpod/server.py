"""
OpenAI-compatible API server for DeepSeek-OCR-2 on vLLM.

This file must live inside the DeepSeek-OCR-2 vLLM directory because it
imports local modules (deepseek_ocr2, process/, config).  The setup.sh
script copies it there automatically.

    DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/server.py

Start:
    python server.py                                    # defaults (port 8000)
    python server.py --port 8000 --gpu-mem 0.9          # custom
    nohup python server.py > /workspace/server.log 2>&1 &  # background

Endpoints:
    POST /v1/chat/completions   OpenAI-compatible, single image
    POST /v1/batch              Batch endpoint — multiple images in one call
    GET  /v1/models             List served model
    GET  /health                Health check

Tested:
    GPU   : RTX 4090 24 GB  (any ≥20 GB works; model uses ~6.3 GB)
    CUDA  : 12.4 / 11.8
    vLLM  : 0.8.5
    Python: 3.11
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch

if torch.version.cuda == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["VLLM_USE_V1"] = "0"

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# ── Local imports (from DeepSeek-OCR2-vllm directory) ───────────────────
from deepseek_ocr2 import DeepseekOCR2ForCausalLM  # noqa: E402
from process.image_process import DeepseekOCR2Processor  # noqa: E402
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # noqa: E402
from config import CROP_MODE  # noqa: E402

# Register the custom architecture with vLLM
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)


# ── CLI args ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeepSeek-OCR-2 vLLM API Server")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-OCR-2", help="HF model ID or local path")
    p.add_argument("--gpu-mem", type=float, default=0.85, help="GPU memory utilisation (0-1)")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-concurrency", type=int, default=64, help="Max concurrent sequences")
    p.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel GPUs")
    return p.parse_args()


# ── Globals (initialised in lifespan) ───────────────────────────────────
llm: LLM | None = None
sampling_params: SamplingParams | None = None
processor: DeepseekOCR2Processor | None = None
args: argparse.Namespace | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, sampling_params, processor, args
    args = parse_args()

    print(f"Loading model {args.model} ...")
    llm = LLM(
        model=args.model,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        dtype="bfloat16",
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        swap_space=0,
        max_num_seqs=args.max_concurrency,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_mem,
        disable_mm_preprocessor_cache=True,
    )

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
        )
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_model_len,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    processor = DeepseekOCR2Processor()
    print(f"Server ready on {args.host}:{args.port}")
    yield


# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(title="DeepSeek-OCR-2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request / Response schemas (OpenAI-compatible subset) ───────────────
class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = 8192
    temperature: float = 0.0
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# ── Helpers ─────────────────────────────────────────────────────────────
def _extract_image_and_prompt(messages: list[Message]) -> tuple[Image.Image | None, str]:
    """Pull the first image and text prompt from the messages."""
    image = None
    prompt_text = ""

    for msg in messages:
        if isinstance(msg.content, str):
            prompt_text = msg.content
            continue
        for part in msg.content:
            if part.type == "text" and part.text:
                prompt_text = part.text
            elif part.type == "image_url" and part.image_url:
                url = part.image_url.url
                if url.startswith("data:"):
                    # data:image/png;base64,<data>
                    b64_data = url.split(",", 1)[1]
                    raw = base64.b64decode(b64_data)
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                    try:
                        image = ImageOps.exif_transpose(image)
                    except Exception:
                        pass

    return image, prompt_text


def _build_prompt(text: str) -> str:
    """Wrap user text into the model's expected prompt format."""
    # The model expects <image>\n<prompt>
    return f"<image>\n{text}"


# ── Endpoint ────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    if llm is None or sampling_params is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    image, prompt_text = _extract_image_and_prompt(req.messages)
    prompt = _build_prompt(prompt_text)

    if image is not None and "<image>" in prompt:
        image_features = processor.tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=CROP_MODE
        )
        request_data = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features},
        }
    else:
        request_data = {"prompt": prompt}

    sp = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        logits_processors=sampling_params.logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    outputs = llm.generate([request_data], sampling_params=sp)
    text = outputs[0].outputs[0].text

    # Strip end-of-sentence token
    text = text.replace("<｜end▁of▁sentence｜>", "")

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model or args.model,
        choices=[Choice(message={"role": "assistant", "content": text})],
        usage=Usage(
            prompt_tokens=len(outputs[0].prompt_token_ids),
            completion_tokens=len(outputs[0].outputs[0].token_ids),
            total_tokens=len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids),
        ),
    )


# ── Batch endpoint ──────────────────────────────────────────────────
class BatchItem(BaseModel):
    image: str  # base64 data-URI: "data:image/png;base64,..."
    prompt: str = "<|grounding|>Convert the document to markdown."


class BatchRequest(BaseModel):
    items: list[BatchItem]
    max_tokens: int = 8192
    temperature: float = 0.0


class BatchResultItem(BaseModel):
    index: int
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BatchResponse(BaseModel):
    results: list[BatchResultItem]
    total_time_ms: int = 0


def _decode_image(data_uri: str) -> Image.Image:
    """Decode a base64 data-URI to a PIL Image."""
    b64_data = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    raw = base64.b64decode(b64_data)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    return image


def _preprocess_single(item: BatchItem) -> dict:
    """Preprocess one batch item into a vLLM request dict."""
    image = _decode_image(item.image)
    prompt = f"<image>\n{item.prompt}"
    image_features = processor.tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=CROP_MODE
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_features},
    }


@app.post("/v1/batch", response_model=BatchResponse)
async def batch_completions(req: BatchRequest):
    """Process multiple images in a single batched vLLM generate() call."""
    if llm is None or sampling_params is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.time()

    # Preprocess images in parallel (CPU-bound resize/tokenize)
    with ThreadPoolExecutor(max_workers=min(len(req.items), 16)) as pool:
        batch_inputs = list(pool.map(_preprocess_single, req.items))

    sp = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        logits_processors=sampling_params.logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    # Single batched generate call — vLLM processes all pages concurrently on GPU
    outputs = llm.generate(batch_inputs, sampling_params=sp)

    results = []
    for idx, output in enumerate(outputs):
        text = output.outputs[0].text.replace("<｜end▁of▁sentence｜>", "")
        results.append(BatchResultItem(
            index=idx,
            text=text,
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.outputs[0].token_ids),
        ))

    return BatchResponse(
        results=results,
        total_time_ms=int((time.time() - t0) * 1000),
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": args.model, "object": "model", "owned_by": "deepseek"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
