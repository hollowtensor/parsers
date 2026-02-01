"""Tests for parser registry."""

import pytest

from parsers.core.registry import ParserRegistry


def test_registered_parsers_exist():
    """Verify that core parsers are registered after import."""
    # Trigger registration by importing methods
    import parsers.methods.docling  # noqa: F401
    import parsers.methods.tesseract  # noqa: F401
    import parsers.methods.vlm.deepseek  # noqa: F401
    import parsers.methods.vlm.dots_ocr  # noqa: F401
    import parsers.methods.vlm.olmocr  # noqa: F401

    assert "docling" in ParserRegistry
    assert "tesseract" in ParserRegistry
    assert "deepseek" in ParserRegistry
    assert "dots_ocr" in ParserRegistry
    assert "olmocr" in ParserRegistry


def test_unknown_parser_raises():
    with pytest.raises(KeyError, match="Unknown component"):
        ParserRegistry.get("nonexistent")
