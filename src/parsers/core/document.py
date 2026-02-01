"""Unified document model for parsed output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box for a detected region."""

    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    """A block of extracted text with optional spatial info."""

    text: str
    bbox: BoundingBox | None = None
    confidence: float | None = None
    block_type: str = "text"  # text, title, table, figure, etc.


class Page(BaseModel):
    """A single page of a parsed document."""

    page_number: int
    width: float | None = None
    height: float | None = None
    blocks: list[TextBlock] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)


class Document(BaseModel):
    """Unified representation of a parsed document."""

    source: str  # file path or identifier
    method: str  # which parser produced this
    pages: list[Page] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def num_pages(self) -> int:
        return len(self.pages)
