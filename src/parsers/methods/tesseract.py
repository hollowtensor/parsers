"""Tesseract OCR parser."""

from __future__ import annotations

from pathlib import Path

from parsers.core.document import Document
from parsers.core.registry import ParserRegistry
from parsers.methods.base import BaseParser


class TesseractParser(BaseParser):
    """Parser using Tesseract OCR engine."""

    name = "tesseract"

    def __init__(self, lang: str = "eng", **kwargs) -> None:
        self.lang = lang
        self.kwargs = kwargs

    def parse(self, file_path: str | Path, **kwargs) -> Document:
        raise NotImplementedError("Tesseract parser not yet implemented")


ParserRegistry.register("tesseract", TesseractParser)
