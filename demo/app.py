"""
Gradio demo for DeepSeek-OCR-2 via a remote vLLM server.

Usage:
    python demo/app.py --server-url https://<pod-id>-8000.proxy.runpod.net/v1
    # or
    DEEPSEEK_SERVER_URL=https://... python demo/app.py
"""

from __future__ import annotations

import argparse
import base64
import os
import re
from io import BytesIO

import fitz
import gradio as gr
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ── Server config ────────────────────────────────────────────────────────
SERVER_URL: str = ""
MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
API_KEY = "EMPTY"
MAX_TOKENS = 8192

# ── Task definitions ─────────────────────────────────────────────────────
TASK_PROMPTS = {
    "Markdown": {
        "prompt": "<|grounding|>Convert the document to markdown.",
        "has_grounding": True,
    },
    "Free OCR": {
        "prompt": "Free OCR.",
        "has_grounding": False,
    },
    "Locate": {
        "prompt": "Locate <|ref|>{query}<|/ref|> in the image.",
        "has_grounding": True,
    },
    "Describe": {
        "prompt": "Describe this image in detail.",
        "has_grounding": False,
    },
    "Custom": {
        "prompt": "",
        "has_grounding": False,
    },
}

# ── Regex for grounding tags ─────────────────────────────────────────────
_GROUNDING_RE = re.compile(
    r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL
)

_STRIP_TOKENS = {"<\uff5cend\u2581of\u2581sentence\uff5c>"}


# ── Server call ──────────────────────────────────────────────────────────
def call_server(image_b64: str, prompt_text: str, mime: str = "image/png") -> str:
    """Send image + prompt to the vLLM server, return raw text."""
    resp = httpx.post(
        f"{SERVER_URL}/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Grounding / bbox helpers ─────────────────────────────────────────────
def extract_grounding_references(text: str):
    return _GROUNDING_RE.findall(text)


def draw_bounding_boxes(
    image: Image.Image, refs: list, extract_images: bool = False
) -> tuple[Image.Image, list[Image.Image]]:
    img_w, img_h = image.size
    img_draw = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
        )
    except OSError:
        font = ImageFont.load_default()

    crops: list[Image.Image] = []
    color_map: dict[str, tuple[int, int, int]] = {}
    rng = np.random.RandomState(42)

    for ref in refs:
        label = ref[1]
        if label not in color_map:
            color_map[label] = (
                int(rng.randint(50, 255)),
                int(rng.randint(50, 255)),
                int(rng.randint(50, 255)),
            )
        color = color_map[label]

        try:
            coords = eval(ref[2])
        except Exception:
            continue

        color_a = color + (60,)

        for box in coords:
            x1 = int(box[0] / 999 * img_w)
            y1 = int(box[1] / 999 * img_h)
            x2 = int(box[2] / 999 * img_w)
            y2 = int(box[3] / 999 * img_h)

            if extract_images and label == "image":
                crops.append(image.crop((x1, y1, x2, y2)))

            width = 5 if label == "title" else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a)

            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))

    img_draw = Image.alpha_composite(img_draw, overlay).convert("RGB")
    return img_draw, crops


def clean_output(text: str, include_images: bool = False) -> str:
    if not text:
        return ""
    for tok in _STRIP_TOKENS:
        text = text.replace(tok, "")

    matches = _GROUNDING_RE.findall(text)
    img_num = 0
    for match in matches:
        if "<|ref|>image<|/ref|>" in match[0]:
            if include_images:
                text = text.replace(
                    match[0], f"\n\n**[Figure {img_num + 1}]**\n\n", 1
                )
                img_num += 1
            else:
                text = text.replace(match[0], "", 1)
        else:
            text = re.sub(
                rf"(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?", "", text
            )

    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    return text.strip()


def embed_images(markdown: str, crops: list[Image.Image]) -> str:
    if not crops:
        return markdown
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(
            f"**[Figure {i + 1}]**",
            f"\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n",
            1,
        )
    return markdown


# ── Image encoding ───────────────────────────────────────────────────────
def encode_pil_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Core processing ──────────────────────────────────────────────────────
def process_image(image: Image.Image, task: str, custom_prompt: str):
    if image is None:
        return "Error: Upload an image", "", "", None, []
    if task in ["Custom", "Locate"] and not custom_prompt.strip():
        return "Please enter a prompt", "", "", None, []

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Build prompt
    if task == "Custom":
        prompt_text = custom_prompt.strip()
        has_grounding = "<|grounding|>" in custom_prompt
    elif task == "Locate":
        prompt_text = f"Locate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
        has_grounding = True
    else:
        prompt_text = TASK_PROMPTS[task]["prompt"]
        has_grounding = TASK_PROMPTS[task]["has_grounding"]

    # Call server
    image_b64 = encode_pil_image(image)
    try:
        result = call_server(image_b64, prompt_text)
    except Exception as e:
        return f"Server error: {e}", "", "", None, []

    if not result:
        return "No text detected", "", "", None, []

    # Parse output
    cleaned = clean_output(result, False)
    markdown = clean_output(result, True)

    img_out = None
    crops: list[Image.Image] = []

    if has_grounding and "<|ref|>" in result:
        refs = extract_grounding_references(result)
        if refs:
            img_out, crops = draw_bounding_boxes(image, refs, True)

    markdown = embed_images(markdown, crops)

    return cleaned, markdown, result, img_out, crops


def process_pdf(
    path: str, task: str, custom_prompt: str, page_num: int
):
    doc = fitz.open(path)
    total_pages = len(doc)
    if page_num < 1 or page_num > total_pages:
        doc.close()
        return (
            f"Invalid page number. PDF has {total_pages} pages.",
            "",
            "",
            None,
            [],
        )
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
    img = Image.open(BytesIO(pix.tobytes("png")))
    doc.close()
    return process_image(img, task, custom_prompt)


def process_file(
    path: str | None, task: str, custom_prompt: str, page_num: int
):
    if not path:
        return "Error: Upload a file", "", "", None, []
    if path.lower().endswith(".pdf"):
        return process_pdf(path, task, custom_prompt, page_num)
    return process_image(Image.open(path), task, custom_prompt)


# ── PDF helpers ──────────────────────────────────────────────────────────
def get_pdf_page_count(file_path: str | None) -> int:
    if not file_path or not file_path.lower().endswith(".pdf"):
        return 1
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count


def load_image(file_path: str | None, page_num: int = 1):
    if not file_path:
        return None
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        page_idx = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False
        )
        img = Image.open(BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    return Image.open(file_path)


def update_page_selector(file_path: str | None):
    if not file_path:
        return gr.update(visible=False)
    if file_path.lower().endswith(".pdf"):
        page_count = get_pdf_page_count(file_path)
        return gr.update(
            visible=True,
            maximum=page_count,
            value=1,
            minimum=1,
            label=f"Select Page (1-{page_count})",
        )
    return gr.update(visible=False)


def toggle_prompt(task: str):
    if task == "Custom":
        return gr.update(
            visible=True,
            label="Custom Prompt",
            placeholder="Add <|grounding|> for bounding boxes",
        )
    if task == "Locate":
        return gr.update(
            visible=True,
            label="Text to Locate",
            placeholder="Enter text to locate",
        )
    return gr.update(visible=False)


def select_boxes(task: str):
    if task == "Locate":
        return gr.update(selected="tab_boxes")
    return gr.update()


# ── Gradio UI ────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    with gr.Blocks(title="DeepSeek-OCR-2 Demo") as app:
        gr.Markdown(
            "# DeepSeek-OCR-2 Demo\n"
            "Convert documents to markdown, extract text, locate content "
            "with bounding boxes. Powered by a remote vLLM server."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Upload Image or PDF",
                    file_types=["image", ".pdf"],
                    type="filepath",
                )
                input_img = gr.Image(
                    label="Input Image", type="pil", height=300
                )
                page_selector = gr.Number(
                    label="Select Page",
                    value=1,
                    minimum=1,
                    step=1,
                    visible=False,
                )
                task = gr.Dropdown(
                    list(TASK_PROMPTS.keys()),
                    value="Markdown",
                    label="Task",
                )
                prompt = gr.Textbox(label="Prompt", lines=2, visible=False)
                btn = gr.Button("Extract", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.Tab("Text", id="tab_text"):
                        text_out = gr.Textbox(
                            lines=20, show_label=False
                        )
                    with gr.Tab("Markdown Preview", id="tab_markdown"):
                        md_out = gr.Markdown("")
                    with gr.Tab("Boxes", id="tab_boxes"):
                        img_out = gr.Image(
                            type="pil", height=500, show_label=False
                        )
                    with gr.Tab("Cropped Images", id="tab_crops"):
                        gallery = gr.Gallery(
                            show_label=False, columns=3, height=400
                        )
                    with gr.Tab("Raw Text", id="tab_raw"):
                        raw_out = gr.Textbox(
                            lines=20, show_label=False
                        )

        with gr.Accordion("Info", open=False):
            gr.Markdown(
                "### Tasks\n"
                "- **Markdown**: Structured markdown with layout detection (bounding boxes)\n"
                "- **Free OCR**: Plain text extraction\n"
                "- **Locate**: Find and highlight specific text/elements\n"
                "- **Describe**: General image description\n"
                "- **Custom**: Your own prompt (add `<|grounding|>` for bounding boxes)\n"
                "\n"
                f"### Server\n"
                f"Connected to: `{SERVER_URL}`"
            )

        # Wire up events
        file_in.change(load_image, [file_in, page_selector], [input_img])
        file_in.change(update_page_selector, [file_in], [page_selector])
        page_selector.change(
            load_image, [file_in, page_selector], [input_img]
        )
        task.change(toggle_prompt, [task], [prompt])
        task.change(select_boxes, [task], [tabs])

        def run(image, file_path, task_val, custom_prompt, page_num):
            if file_path:
                return process_file(
                    file_path, task_val, custom_prompt, int(page_num)
                )
            if image is not None:
                return process_image(image, task_val, custom_prompt)
            return "Error: Upload a file or image", "", "", None, []

        submit = btn.click(
            run,
            [input_img, file_in, task, prompt, page_selector],
            [text_out, md_out, raw_out, img_out, gallery],
        )
        submit.then(select_boxes, [task], [tabs])

    return app


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    global SERVER_URL

    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 Gradio Demo")
    parser.add_argument(
        "--server-url",
        default=os.environ.get(
            "DEEPSEEK_SERVER_URL", "http://localhost:8000/v1"
        ),
        help="vLLM server base URL (e.g. https://<pod>-8000.proxy.runpod.net/v1)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    SERVER_URL = args.server_url.rstrip("/")
    print(f"Server URL: {SERVER_URL}")

    # Health check
    try:
        health_url = SERVER_URL.rsplit("/v1", 1)[0] + "/health"
        resp = httpx.get(health_url, timeout=10.0)
        print(f"Server health: {resp.json()}")
    except Exception as e:
        print(f"Warning: Could not reach server health endpoint: {e}")

    app = build_app()
    app.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
