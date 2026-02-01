"""DeepSeek OCR parser using DeepSeek vision-language model."""

from __future__ import annotations

from pathlib import Path

from parsers.core.document import Document
from parsers.core.registry import ParserRegistry
from parsers.methods.vlm.base import BaseVLMParser


class DeepSeekOCRParser(BaseVLMParser):
    """Parser using DeepSeek's vision-language model for OCR."""

    name = "deepseek"

    def parse(self, file_path: str | Path, **kwargs) -> Document:
        raise NotImplementedError("DeepSeek OCR parser not yet implemented")


ParserRegistry.register("deepseek", DeepSeekOCRParser)
