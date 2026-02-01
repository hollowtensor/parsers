"""dots.ocr parser using vision-language model."""

from __future__ import annotations

from pathlib import Path

from parsers.core.document import Document
from parsers.core.registry import ParserRegistry
from parsers.methods.vlm.base import BaseVLMParser


class DotsOCRParser(BaseVLMParser):
    """Parser using dots.ocr vision-language model."""

    name = "dots_ocr"

    def parse(self, file_path: str | Path, **kwargs) -> Document:
        raise NotImplementedError("dots.ocr parser not yet implemented")


ParserRegistry.register("dots_ocr", DotsOCRParser)
