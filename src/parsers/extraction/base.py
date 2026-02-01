"""Base class for structured extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from parsers.core.document import Document

T = TypeVar("T", bound=BaseModel)


class BaseExtractor(ABC):
    """Abstract base for structured extractors.

    Extractors take a parsed Document and produce structured output
    conforming to a Pydantic schema.
    """

    name: str

    @abstractmethod
    def extract(self, document: Document, schema: type[T], **kwargs) -> T:
        """Extract structured data from a document according to a schema."""
        ...

    async def aextract(self, document: Document, schema: type[T], **kwargs) -> T:
        """Async variant. Defaults to sync implementation."""
        return self.extract(document, schema, **kwargs)
