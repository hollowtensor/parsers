"""
Gradio demo for LightOnOCR-2 via a remote vLLM server.

Mirrors the official LightOnOCR-2-1B-Demo UI but calls a remote vLLM
endpoint instead of loading the model locally.

Usage:
    python demo/lightonocr_app.py --server-url https://<pod-id>-8000.proxy.runpod.net/v1
    LIGHTONOCR_SERVER_URL=https://... python demo/lightonocr_app.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from io import BytesIO

import gradio as gr
import httpx
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

# ── Server config (set at startup) ───────────────────────────────────────
SERVER_URL: str = ""
API_KEY = "not-needed"

# ── Streaming config ─────────────────────────────────────────────────────
STREAM_YIELD_INTERVAL = 0.5  # seconds between UI updates during streaming

# ── Model registry (matches the official demo) ──────────────────────────
MODEL_REGISTRY = {
    "LightOnOCR-2-1B (Best OCR)": {
        "model_id": "lightonai/LightOnOCR-2-1B",
        "has_bbox": False,
        "description": "Best overall OCR performance",
    },
    "LightOnOCR-2-1B-bbox (Best Bbox)": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox",
        "has_bbox": True,
        "description": "Best bounding box detection",
    },
    "LightOnOCR-2-1B-base": {
        "model_id": "lightonai/LightOnOCR-2-1B-base",
        "has_bbox": False,
        "description": "Base OCR model",
    },
    "LightOnOCR-2-1B-bbox-base": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox-base",
        "has_bbox": True,
        "description": "Base bounding box model",
    },
    "LightOnOCR-2-1B-ocr-soup": {
        "model_id": "lightonai/LightOnOCR-2-1B-ocr-soup",
        "has_bbox": False,
        "description": "OCR soup variant",
    },
    "LightOnOCR-2-1B-bbox-soup": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox-soup",
        "has_bbox": True,
        "description": "Bounding box soup variant",
    },
}

DEFAULT_MODEL = "LightOnOCR-2-1B (Best OCR)"

# ── Bbox pattern: ![image](image_N.png)x1,y1,x2,y2 ─────────────────────
BBOX_PATTERN = re.compile(
    r"!\[image\]\((image_\d+\.png)\)\s*(\d+),(\d+),(\d+),(\d+)"
)

# ── PDF rendering (200 DPI as recommended by LightOnOCR docs) ────────────
PDF_SCALE = 2.77  # 200/72
MAX_RESOLUTION = 1540


# ── Server call (supports streaming) ─────────────────────────────────────
def extract_text_via_vllm(
    image: Image.Image,
    model_name: str,
    temperature: float = 0.2,
    stream: bool = False,
    max_tokens: int = 2048,
):
    """Extract text from image using vLLM endpoint. Yields text progressively."""
    config = MODEL_REGISTRY.get(model_name)
    if config is None:
        yield f"Unknown model: {model_name}"
        return

    model_id = config["model_id"]

    # Encode image to base64 data URI
    buf = BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    image_uri = f"data:image/png;base64,{image_b64}"

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else 0.0,
        "top_p": 0.9,
        "stream": stream,
    }

    headers = {"Authorization": f"Bearer {API_KEY}"}

    if stream:
        full_text = ""
        last_yield = time.time()
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{SERVER_URL}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        if time.time() - last_yield > STREAM_YIELD_INTERVAL:
                            yield clean_output(full_text)
                            last_yield = time.time()
        yield clean_output(full_text)
    else:
        resp = httpx.post(
            f"{SERVER_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=300.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        yield clean_output(text)


# ── Output cleaning ──────────────────────────────────────────────────────
def clean_output(text: str) -> str:
    """Remove chat template artifacts from output."""
    if not text:
        return ""
    markers = ["system", "user", "assistant"]
    lines = text.split("\n")
    cleaned = [l for l in lines if l.strip().lower() not in markers]
    result = "\n".join(cleaned).strip()

    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            result = parts[1].strip()
    return result


# ── Bbox helpers ─────────────────────────────────────────────────────────
def parse_bbox_output(text: str):
    """Parse bbox output. Returns (cleaned_text, list of detections)."""
    detections = []
    for m in BBOX_PATTERN.finditer(text):
        ref, x1, y1, x2, y2 = m.groups()
        detections.append(
            {"ref": ref, "coords": (int(x1), int(y1), int(x2), int(y2))}
        )
    cleaned = BBOX_PATTERN.sub(r"![image](\1)", text)
    return cleaned, detections


def crop_from_bbox(
    image: Image.Image, bbox: dict, padding: int = 5
) -> Image.Image:
    """Crop region from image. Coords are normalized 0-1000."""
    w, h = image.size
    x1, y1, x2, y2 = bbox["coords"]
    px1 = max(0, int(x1 * w / 1000) - padding)
    py1 = max(0, int(y1 * h / 1000) - padding)
    px2 = min(w, int(x2 * w / 1000) + padding)
    py2 = min(h, int(y2 * h / 1000) + padding)
    return image.crop((px1, py1, px2, py2))


def render_bbox_with_crops(raw_text: str, image: Image.Image) -> str:
    """Replace markdown image placeholders with actual cropped base64 images."""
    cleaned, detections = parse_bbox_output(raw_text)
    for bbox in detections:
        try:
            cropped = crop_from_bbox(image, bbox)
            buf = BytesIO()
            cropped.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            cleaned = cleaned.replace(
                f"![image]({bbox['ref']})",
                f"![Cropped region](data:image/png;base64,{b64})",
            )
        except Exception:
            continue
    return cleaned


# ── PDF helpers ──────────────────────────────────────────────────────────
def render_pdf_page(
    page, max_res: int = MAX_RESOLUTION, scale: float = PDF_SCALE
):
    w, h = page.get_size()
    pw, ph = w * scale, h * scale
    factor = min(1, max_res / pw, max_res / ph)
    return page.render(scale=scale * factor, rev_byteorder=True).to_pil()


def process_pdf(pdf_path: str, page_num: int = 1):
    """Extract a specific page from PDF."""
    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    idx = max(0, min(int(page_num) - 1, total - 1))
    img = render_pdf_page(pdf[idx])
    pdf.close()
    return img, total, idx + 1


# ── Core processing (streaming generator) ────────────────────────────────
def process_input(
    file_input,
    model_name,
    temperature,
    page_num,
    enable_streaming,
    max_tokens,
):
    """Process uploaded file and extract text with optional streaming."""
    if file_input is None:
        yield (
            "*Upload an image or PDF first.*",
            "",
            "",
            None,
            gr.update(),
        )
        return

    file_path = file_input if isinstance(file_input, str) else file_input.name
    page_info = ""
    image = None

    # Load image
    if file_path.lower().endswith(".pdf"):
        try:
            image, total, actual = process_pdf(file_path, int(page_num))
            page_info = f"Page {actual} of {total}"
        except Exception as e:
            yield f"Error: {e}", "", "", None, gr.update()
            return
    else:
        try:
            image = Image.open(file_path).convert("RGB")
            page_info = "Image loaded"
        except Exception as e:
            yield f"Error: {e}", "", "", None, gr.update()
            return

    model_info = MODEL_REGISTRY.get(model_name, {})
    has_bbox = model_info.get("has_bbox", False)

    try:
        for extracted_text in extract_text_via_vllm(
            image,
            model_name,
            temperature,
            stream=enable_streaming,
            max_tokens=int(max_tokens),
        ):
            if has_bbox:
                rendered = render_bbox_with_crops(extracted_text, image)
            else:
                rendered = extracted_text

            yield rendered, extracted_text, page_info, image, gr.update()

    except Exception as e:
        yield f"Server error: {e}", str(e), page_info, image, gr.update()


# ── UI helpers ───────────────────────────────────────────────────────────
def update_slider_and_preview(file_input):
    if file_input is None:
        return gr.update(maximum=20, value=1), None
    file_path = file_input if isinstance(file_input, str) else file_input.name
    if file_path.lower().endswith(".pdf"):
        try:
            pdf = pdfium.PdfDocument(file_path)
            total = len(pdf)
            preview = pdf[0].render(scale=2).to_pil()
            pdf.close()
            return gr.update(maximum=total, value=1), preview
        except Exception:
            return gr.update(maximum=20, value=1), None
    else:
        try:
            return gr.update(maximum=1, value=1), Image.open(file_path)
        except Exception:
            return gr.update(maximum=1, value=1), None


def get_model_info_text(model_name: str) -> str:
    info = MODEL_REGISTRY.get(model_name, {})
    has_bbox = (
        "Yes -- will show cropped regions inline"
        if info.get("has_bbox", False)
        else "No"
    )
    return (
        f"**Description:** {info.get('description', 'N/A')}\n"
        f"**Bounding Box Detection:** {has_bbox}"
    )


# ── Gradio UI ────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    with gr.Blocks(title="LightOnOCR-2 Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# LightOnOCR-2 -- OCR Demo\n\n"
            "State-of-the-art OCR on OmniDocBench. ~9x smaller and faster "
            "than competitors. Handles tables, forms, math, multi-column "
            "layouts.\n\n"
            "**How to use:** Select a model -> Upload image/PDF -> "
            "Click \"Extract Text\"\n\n"
            f"**Server:** `{SERVER_URL}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    value=DEFAULT_MODEL,
                    label="Model",
                    info="Select OCR model variant",
                )
                model_info = gr.Markdown(
                    value=get_model_info_text(DEFAULT_MODEL),
                    label="Model Info",
                )
                file_input = gr.File(
                    label="Upload Image or PDF",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                rendered_image = gr.Image(
                    label="Preview",
                    type="pil",
                    height=400,
                    interactive=False,
                )
                num_pages = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    label="PDF: Page Number",
                    info="Select which page to extract",
                )
                page_info = gr.Textbox(
                    label="Processing Info", value="", interactive=False
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.05,
                    label="Temperature",
                    info="0.0 = deterministic, higher = more varied",
                )
                enable_streaming = gr.Checkbox(
                    label="Enable Streaming",
                    value=True,
                    info="Show text progressively as it's generated",
                )
                max_output_tokens = gr.Slider(
                    minimum=256,
                    maximum=8192,
                    value=2048,
                    step=256,
                    label="Max Output Tokens",
                    info="Maximum number of tokens to generate",
                )
                submit_btn = gr.Button(
                    "Extract Text", variant="primary", size="lg"
                )
                clear_btn = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=2):
                output_text = gr.Markdown(
                    label="Extracted Text (Rendered)",
                    value="*Extracted text will appear here...*",
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )

        with gr.Row():
            with gr.Column():
                raw_output = gr.Textbox(
                    label="Raw Markdown Output",
                    placeholder="Raw text will appear here...",
                    lines=20,
                    max_lines=30,
                )

        with gr.Accordion("Info", open=False):
            gr.Markdown(
                "### About LightOnOCR-2\n"
                "- **1B parameters**, end-to-end differentiable\n"
                "- 3.3x faster than Chandra, 1.7x faster than OlmOCR\n"
                "- <$0.01 per 1,000 pages on H100\n"
                "- Supports tables, forms, math, multi-column layouts\n"
                "- **Bbox variants** detect image regions with coordinates\n"
                "\n"
                "### Tips\n"
                "- PDFs rendered at 200 DPI (longest edge capped at 1540px)\n"
                "- Temperature 0.2 works well for most documents\n"
                "- Use bbox model variants to see image regions cropped inline"
            )

        # ── Event handlers ───────────────────────────────────────────────
        submit_btn.click(
            fn=process_input,
            inputs=[
                file_input,
                model_selector,
                temperature,
                num_pages,
                enable_streaming,
                max_output_tokens,
            ],
            outputs=[
                output_text,
                raw_output,
                page_info,
                rendered_image,
                num_pages,
            ],
        )

        file_input.change(
            fn=update_slider_and_preview,
            inputs=[file_input],
            outputs=[num_pages, rendered_image],
        )

        model_selector.change(
            fn=get_model_info_text,
            inputs=[model_selector],
            outputs=[model_info],
        )

        clear_btn.click(
            fn=lambda: (
                None,
                DEFAULT_MODEL,
                get_model_info_text(DEFAULT_MODEL),
                "*Extracted text will appear here...*",
                "",
                "",
                None,
                1,
                2048,
            ),
            outputs=[
                file_input,
                model_selector,
                model_info,
                output_text,
                raw_output,
                page_info,
                rendered_image,
                num_pages,
                max_output_tokens,
            ],
        )

    return app


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    global SERVER_URL

    parser = argparse.ArgumentParser(description="LightOnOCR-2 Gradio Demo")
    parser.add_argument(
        "--server-url",
        default=os.environ.get(
            "LIGHTONOCR_SERVER_URL", "http://localhost:8000/v1"
        ),
        help="vLLM server base URL",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    SERVER_URL = args.server_url.rstrip("/")
    print(f"Server URL: {SERVER_URL}")

    # Health check
    try:
        health_url = SERVER_URL.rsplit("/v1", 1)[0] + "/health"
        resp = httpx.get(health_url, timeout=10.0)
        print(f"Server health: {resp.json()}")
    except Exception as e:
        print(f"Warning: Could not reach server: {e}")

    app = build_app()
    app.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
