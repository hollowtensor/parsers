"""Document parsing service with pluggable methods and structured extraction."""

from parsers.core.document import Document, Page
from parsers.core.registry import ParserRegistry, ExtractorRegistry

__all__ = [
    "Document",
    "Page",
    "ParserRegistry",
    "ExtractorRegistry",
]
