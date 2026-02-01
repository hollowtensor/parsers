"""OLM OCR parser using vision-language model."""

from __future__ import annotations

from pathlib import Path

from parsers.core.document import Document
from parsers.core.registry import ParserRegistry
from parsers.methods.vlm.base import BaseVLMParser


class OLMOCRParser(BaseVLMParser):
    """Parser using OLM OCR vision-language model."""

    name = "olmocr"

    def parse(self, file_path: str | Path, **kwargs) -> Document:
        raise NotImplementedError("OLM OCR parser not yet implemented")


ParserRegistry.register("olmocr", OLMOCRParser)
