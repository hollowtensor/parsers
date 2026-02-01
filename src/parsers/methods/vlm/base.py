"""Base class for vision-language model (VLM) based parsers."""

from __future__ import annotations

import base64
from abc import abstractmethod
from pathlib import Path

from parsers.core.document import Document
from parsers.methods.base import BaseParser


class BaseVLMParser(BaseParser):
    """Base for VLM-based OCR/parsing methods.

    These parsers use vision-language models to extract text
    and structure from document images.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    @abstractmethod
    def parse(self, file_path: str | Path, **kwargs) -> Document: ...

    def _encode_image(self, file_path: str | Path) -> str:
        """Encode an image file to base64."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_bytes(self, data: bytes) -> str:
        """Encode raw bytes to base64."""
        return base64.b64encode(data).decode("utf-8")

    def _get_mime_type(self, file_path: str | Path) -> str:
        """Get MIME type from file extension."""
        ext = Path(file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".bmp": "image/bmp",
            ".pdf": "application/pdf",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _pdf_to_images(self, file_path: str | Path, dpi: int = 144) -> list[bytes]:
        """Convert PDF pages to PNG images. Returns list of PNG bytes per page."""
        import fitz  # pymupdf

        doc = fitz.open(str(file_path))
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(pix.tobytes("png"))
        doc.close()
        return images

    def _get_page_dimensions(self, file_path: str | Path) -> list[tuple[float, float]]:
        """Get (width, height) for each page in a PDF."""
        import fitz

        doc = fitz.open(str(file_path))
        dims = [(page.rect.width, page.rect.height) for page in doc]
        doc.close()
        return dims
