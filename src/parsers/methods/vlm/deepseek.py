"""DeepSeek OCR-2 parser — client for a remote vLLM server."""

from __future__ import annotations

import re
from pathlib import Path

from parsers.core.document import BoundingBox, Document, Page, TextBlock
from parsers.core.registry import ParserRegistry
from parsers.methods.vlm.base import BaseVLMParser

# Prompt templates from the official DeepSeek-OCR-2 repo + HF demo
PROMPTS = {
    "document": "<|grounding|>Convert the document to markdown.",
    "ocr": "Free OCR.",
    "describe": "Describe this image in detail.",
}

# Locate mode is special — the query text is interpolated at runtime
_LOCATE_TEMPLATE = "Locate <|ref|>{query}<|/ref|> in the image."

# Regex: <|ref|>label<|/ref|><|det|>coords<|/det|> followed by content until next tag or end
_GROUNDING_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.DOTALL
)

# Splits text into (tag, content) pairs — tag comes before its content
_BLOCK_SPLIT_RE = re.compile(
    r"(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)", re.DOTALL
)

# Tokens the model may emit that we want to strip
_STRIP_TOKENS = {"<｜end▁of▁sentence｜>"}


def _parse_bbox(
    coords_raw: str, page_width: float | None, page_height: float | None
) -> BoundingBox | None:
    """Parse a coordinate string like [[x1,y1,x2,y2]] into a BoundingBox."""
    try:
        coord_lists = eval(coords_raw)
    except Exception:
        return None
    if not coord_lists or len(coord_lists[0]) != 4:
        return None
    x0, y0, x1, y1 = coord_lists[0]
    return BoundingBox(
        x0=x0 / 999 * page_width if page_width else x0,
        y0=y0 / 999 * page_height if page_height else y0,
        x1=x1 / 999 * page_width if page_width else x1,
        y1=y1 / 999 * page_height if page_height else y1,
    )


def _clean(text: str) -> str:
    for tok in _STRIP_TOKENS:
        text = text.replace(tok, "")
    # LaTeX artifacts the model sometimes emits
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    return text.strip()


def _parse_grounding(raw_text: str, page_width: float | None, page_height: float | None):
    """Parse DeepSeek-OCR-2 grounding output into blocks + cleaned markdown.

    Output format: each grounding tag precedes its content:
        <|ref|>type<|/ref|><|det|>[[coords]]<|/det|>
        content text here ...

    Returns (cleaned_text, list[TextBlock]).
    """
    raw_text = _clean(raw_text)
    parts = _BLOCK_SPLIT_RE.split(raw_text)

    blocks: list[TextBlock] = []
    cleaned_parts: list[str] = []
    pending_label: str | None = None
    pending_bbox: BoundingBox | None = None

    for part in parts:
        tag_match = _GROUNDING_RE.fullmatch(part)
        if tag_match:
            # This is a grounding tag — stash label + bbox for the next content part
            pending_label = tag_match.group(1).strip()
            pending_bbox = _parse_bbox(tag_match.group(2).strip(), page_width, page_height)
        else:
            # This is content text
            content = part.strip()
            if pending_label is not None and content:
                block_type = pending_label if pending_label != "image" else "figure"
                blocks.append(TextBlock(
                    text=content,
                    bbox=pending_bbox,
                    block_type=block_type,
                ))
                cleaned_parts.append(content)
                pending_label = None
                pending_bbox = None
            elif content:
                # Text before any grounding tag
                blocks.append(TextBlock(text=content))
                cleaned_parts.append(content)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n\n".join(cleaned_parts)).strip()
    return cleaned, blocks


class DeepSeekOCRParser(BaseVLMParser):
    """Client parser for DeepSeek-OCR-2 running on a remote vLLM server.

    Expects a vLLM server with the OpenAI-compatible API exposed, e.g.:
        vllm serve deepseek-ai/DeepSeek-OCR-2 --trust-remote-code

    Usage:
        parser = DeepSeekOCRParser(base_url="http://<runpod-ip>:8000/v1")
        doc = parser.parse("invoice.pdf")
    """

    name = "deepseek"

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "deepseek-ai/DeepSeek-OCR-2",
        prompt_mode: str = "document",
        locate_query: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        pdf_dpi: int = 150,
        batch_size: int = 8,
        retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.prompt_mode = prompt_mode
        self.locate_query = locate_query
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.pdf_dpi = pdf_dpi
        self.batch_size = batch_size
        self.retries = retries
        self.retry_delay = retry_delay

    @property
    def _prompt_text(self) -> str:
        if self.prompt_mode == "locate":
            query = self.locate_query or "text"
            return _LOCATE_TEMPLATE.format(query=query)
        return PROMPTS.get(self.prompt_mode, PROMPTS["document"])

    def _build_message(self, image_b64: str, mime: str = "image/png") -> list[dict]:
        """Build an OpenAI-compatible chat message with an image."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {"type": "text", "text": self._prompt_text},
                ],
            }
        ]

    def _call_server(self, image_b64: str, mime: str = "image/png") -> str:
        """Send a single image to the vLLM server and return raw text.

        Retries on transient errors (SSL, connection, timeout).
        """
        import time
        import httpx

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": self._build_message(image_b64, mime),
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=300.0,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_delay * attempt)
        raise last_exc  # type: ignore[misc]

    def _call_batch(self, images_b64: list[str], mime: str = "image/png") -> list[str]:
        """Send images in chunked batch requests. Returns list of raw texts in order."""
        import httpx

        all_results: list[str] = []
        for i in range(0, len(images_b64), self.batch_size):
            chunk = images_b64[i : i + self.batch_size]
            items = [
                {"image": f"data:{mime};base64,{b64}", "prompt": self._prompt_text}
                for b64 in chunk
            ]
            resp = httpx.post(
                f"{self.base_url}/batch",
                json={
                    "items": items,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=600.0,
            )
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(r["text"] for r in data["results"])
        return all_results

    async def _acall_server(self, image_b64: str, mime: str = "image/png") -> str:
        """Async variant of _call_server with retries."""
        import asyncio
        import httpx

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": self._build_message(image_b64, mime),
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                        },
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self.retries:
                    await asyncio.sleep(self.retry_delay * attempt)
        raise last_exc  # type: ignore[misc]

    # ── Sync ────────────────────────────────────────────────────────────

    def parse(self, file_path: str | Path, **kwargs) -> Document:
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".pdf":
            return self._parse_pdf(file_path, **kwargs)
        return self._parse_image(file_path, **kwargs)

    def _parse_image(self, file_path: Path, **kwargs) -> Document:
        b64 = self._encode_image(file_path)
        mime = self._get_mime_type(file_path)
        raw = self._call_server(b64, mime)

        cleaned, blocks = _parse_grounding(raw, None, None)

        # If no grounding blocks, treat entire response as a single text block
        if not blocks:
            blocks = [TextBlock(text=cleaned)]

        page = Page(page_number=1, blocks=blocks)
        return Document(
            source=str(file_path),
            method=self.name,
            pages=[page],
            metadata={"raw_response": raw, "prompt_mode": self.prompt_mode},
        )

    def _parse_pdf(self, file_path: Path, **kwargs) -> Document:
        page_images = self._pdf_to_images(file_path, dpi=self.pdf_dpi)
        dims = self._get_page_dimensions(file_path)

        pages: list[Page] = []
        for idx, (img_bytes, (pw, ph)) in enumerate(zip(page_images, dims)):
            b64 = self._encode_bytes(img_bytes)
            raw = self._call_server(b64, "image/png")
            cleaned, blocks = _parse_grounding(raw, pw, ph)
            if not blocks:
                blocks = [TextBlock(text=cleaned)]
            pages.append(Page(page_number=idx + 1, width=pw, height=ph, blocks=blocks))

        return Document(
            source=str(file_path),
            method=self.name,
            pages=pages,
            metadata={"prompt_mode": self.prompt_mode},
        )

    # ── Async ───────────────────────────────────────────────────────────

    async def aparse(self, file_path: str | Path, **kwargs) -> Document:
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pdf":
            return await self._aparse_pdf(file_path, **kwargs)
        return await self._aparse_image(file_path, **kwargs)

    async def _aparse_image(self, file_path: Path, **kwargs) -> Document:
        b64 = self._encode_image(file_path)
        mime = self._get_mime_type(file_path)
        raw = await self._acall_server(b64, mime)

        cleaned, blocks = _parse_grounding(raw, None, None)
        if not blocks:
            blocks = [TextBlock(text=cleaned)]

        page = Page(page_number=1, blocks=blocks)
        return Document(
            source=str(file_path),
            method=self.name,
            pages=[page],
            metadata={"raw_response": raw, "prompt_mode": self.prompt_mode},
        )

    async def _aparse_pdf(self, file_path: Path, **kwargs) -> Document:
        import asyncio

        page_images = self._pdf_to_images(file_path, dpi=self.pdf_dpi)
        dims = self._get_page_dimensions(file_path)

        async def _do_page(idx: int, img_bytes: bytes, pw: float, ph: float) -> Page:
            b64 = self._encode_bytes(img_bytes)
            raw = await self._acall_server(b64, "image/png")
            cleaned, blocks = _parse_grounding(raw, pw, ph)
            if not blocks:
                blocks = [TextBlock(text=cleaned)]
            return Page(page_number=idx + 1, width=pw, height=ph, blocks=blocks)

        tasks = [
            _do_page(i, img, pw, ph)
            for i, (img, (pw, ph)) in enumerate(zip(page_images, dims))
        ]
        pages = await asyncio.gather(*tasks)
        return Document(
            source=str(file_path),
            method=self.name,
            pages=list(pages),
            metadata={"prompt_mode": self.prompt_mode},
        )


ParserRegistry.register("deepseek", DeepSeekOCRParser)
