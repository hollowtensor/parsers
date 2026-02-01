"""Tests for DeepSeek OCR-2 parser — unit tests (no server required)."""

from parsers.methods.vlm.deepseek import _parse_grounding, PROMPTS


def test_parse_grounding_extracts_blocks():
    raw = (
        "# Title\n"
        "<|ref|>title<|/ref|><|det|>[[10,20,500,80]]<|/det|>\n"
        "Some body text here.\n"
        "<|ref|>text<|/ref|><|det|>[[10,100,500,400]]<|/det|>\n"
        "<|ref|>image<|/ref|><|det|>[[50,420,450,700]]<|/det|>\n"
    )
    cleaned, blocks = _parse_grounding(raw, page_width=1000, page_height=1000)

    assert len(blocks) == 3
    assert blocks[0].block_type == "title"
    assert blocks[1].block_type == "text"
    assert blocks[2].block_type == "figure"

    # Coords should be scaled from 0-999 to page dimensions
    assert blocks[0].bbox.x0 == 10 / 999 * 1000
    assert blocks[0].bbox.y0 == 20 / 999 * 1000

    # Cleaned text should have grounding tags removed
    assert "<|ref|>" not in cleaned
    assert "<|det|>" not in cleaned
    assert "Title" in cleaned
    assert "Some body text here." in cleaned


def test_parse_grounding_no_tags():
    raw = "Just plain markdown text.\n\nAnother paragraph."
    cleaned, blocks = _parse_grounding(raw, None, None)

    assert blocks == []
    assert "Just plain markdown text." in cleaned


def test_parse_grounding_strips_eos_token():
    raw = "Some text<｜end▁of▁sentence｜>"
    cleaned, blocks = _parse_grounding(raw, None, None)

    assert "<｜end▁of▁sentence｜>" not in cleaned
    assert "Some text" in cleaned


def test_prompt_modes():
    assert "grounding" in PROMPTS["document"]
    assert "Free OCR" in PROMPTS["ocr"]
