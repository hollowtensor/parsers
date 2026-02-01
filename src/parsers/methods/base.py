"""Base class for all parsing methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from parsers.core.document import Document


class BaseParser(ABC):
    """Abstract base for document parsers.

    All parsing methods (Docling, Tesseract, VLM-based, etc.) must
    implement this interface.
    """

    name: str  # unique identifier for this parser

    @abstractmethod
    def parse(self, file_path: str | Path, **kwargs) -> Document:
        """Parse a document and return a unified Document."""
        ...

    async def aparse(self, file_path: str | Path, **kwargs) -> Document:
        """Async variant. Defaults to sync implementation."""
        return self.parse(file_path, **kwargs)

    def supports(self, file_path: str | Path) -> bool:
        """Check if this parser supports the given file type."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions

    @property
    def supported_extensions(self) -> set[str]:
        """File extensions this parser can handle."""
        return {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
