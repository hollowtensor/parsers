"""DSPy-based structured extraction with programmatic LLM pipelines."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from parsers.core.document import Document
from parsers.core.registry import ExtractorRegistry
from parsers.extraction.base import BaseExtractor

T = TypeVar("T", bound=BaseModel)


class DSPyExtractor(BaseExtractor):
    """Extract structured data using DSPy pipelines with Pydantic schema enforcement."""

    name = "dspy"

    def __init__(self, model: str | None = None, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def extract(self, document: Document, schema: type[T], **kwargs) -> T:
        raise NotImplementedError("DSPy extractor not yet implemented")


ExtractorRegistry.register("dspy", DSPyExtractor)
