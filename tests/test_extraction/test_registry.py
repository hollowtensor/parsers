"""Tests for extractor registry."""

import pytest

from parsers.core.registry import ExtractorRegistry


def test_registered_extractors_exist():
    """Verify that core extractors are registered after import."""
    import parsers.extraction.llm  # noqa: F401
    import parsers.extraction.dspy_extractor  # noqa: F401

    assert "llm" in ExtractorRegistry
    assert "dspy" in ExtractorRegistry


def test_unknown_extractor_raises():
    with pytest.raises(KeyError, match="Unknown component"):
        ExtractorRegistry.get("nonexistent")
