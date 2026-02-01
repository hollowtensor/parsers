"""Docling-based document parser."""

from __future__ import annotations

from pathlib import Path

from parsers.core.document import Document
from parsers.core.registry import ParserRegistry
from parsers.methods.base import BaseParser


class DoclingParser(BaseParser):
    """Parser using IBM Docling for layout-aware document understanding."""

    name = "docling"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def parse(self, file_path: str | Path, **kwargs) -> Document:
        raise NotImplementedError("Docling parser not yet implemented")


ParserRegistry.register("docling", DoclingParser)
