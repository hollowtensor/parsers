"""LLM-based structured extraction."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from parsers.core.document import Document
from parsers.core.registry import ExtractorRegistry
from parsers.extraction.base import BaseExtractor

T = TypeVar("T", bound=BaseModel)


class LLMExtractor(BaseExtractor):
    """Extract structured data using LLM prompting with schema enforcement."""

    name = "llm"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def extract(self, document: Document, schema: type[T], **kwargs) -> T:
        raise NotImplementedError("LLM extractor not yet implemented")


ExtractorRegistry.register("llm", LLMExtractor)
